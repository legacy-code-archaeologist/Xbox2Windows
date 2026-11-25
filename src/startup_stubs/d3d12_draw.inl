// Experimental (docs/startup-runtime.md): draws through translated pipelines.
// Included inside d3d12_xs_device.cpp's anonymous namespace, after the Pass helpers.
//
// A machine-code pipeline is looked up by the hash of its two code blobs and its root signature in
// X2W_SHADER_CACHE: <key>.txt (manifest), <key>.vs.cso and <key>.ps.cso, produced offline by
// out/gamerun/gfx2hlsl.py. A miss writes req_<key>.{a,b,root} there for the next offline pass.
// The translated shaders use their own root signature: [0] 32-bit constants ("entry" values such
// as the base vertex), an optional SRV table and an optional sampler table. Each draw copies what the title bound (vertex streams,
// buffers named by the hardware descriptors in its tables, textures behind view tokens) into an
// upload ring / a private descriptor heap.

struct Translation {
    struct Srv { int kind, source, a, b; };  // kind 0 buffer, 1 2D texture, 2 2D array, 3 3D texture, 4 1D texture; source 0 table(param a, byte b), 1 vertex stream a, 2 root parameter a
    enum UdKind { rect, base_vertex, start_instance, root_constant, stride, mrt_bits };
    struct Ud { int kind, a, b; };
    int state = 0;                           // 1 ready, 2 unavailable
    ULONGLONG retry_manifest_at = 0;         // Only an absent manifest can become ready on a later draw.
    std::vector<Srv> srvs;
    std::vector<Ud> uds;
    x2w_startup::TranslatedSamplers samplers;
    x2w_startup::ExportWriteMasks write_masks;  // manifest 'wmask': channels each export leaves unwritten (#1674)
    ID3DBlob* vs = nullptr;
    ID3DBlob* ps = nullptr;
    ID3D12RootSignature* root = nullptr;
    std::unordered_map<u64, ID3D12PipelineState*> pipelines;  // by target formats
    u64 key = 0;
    LONG logged_draws = 0, logged_bindings = 0, logged_large = 0;  // diagnostics budget per translation
};

// Per title pipeline: its translation and the blend state found in its state block (if any).
struct BoundPipeline {
    Translation* translation = nullptr;
    bool has_blend = false;
    D3D12_BLEND_DESC blend{};
    bool has_depth_stencil = false;
    D3D12_DEPTH_STENCIL_DESC depth_stencil{};
    bool depth_from_registers = false;  // X2W_DEPTH_STATE: depth_stencil was decoded from state block A (slot 57)
    bool has_stencil_ref = false;       // ... and its stencil test is on: stencil_ref goes on the list with each draw
    std::uint8_t stencil_ref = 0;
    DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN;
    std::unordered_map<u64, ID3D12PipelineState*> pipelines;  // by target formats
};

// Looks for a desktop-layout D3D12_BLEND_DESC inside a state block: two flags, then eight render
// target entries whose fields are all inside their enumerations.
bool find_blend(const std::vector<unsigned char>& block, D3D12_BLEND_DESC* out, std::size_t* where)
{
    for (std::size_t at = 0; at + sizeof(D3D12_BLEND_DESC) <= block.size(); at += 4) {
        D3D12_BLEND_DESC d;
        std::memcpy(&d, block.data() + at, sizeof(d));
        bool ok = (d.AlphaToCoverageEnable == 0 || d.AlphaToCoverageEnable == 1) && (d.IndependentBlendEnable == 0 || d.IndependentBlendEnable == 1);
        for (int i = 0; ok && i < 8; ++i) {
            const auto& t = d.RenderTarget[i];
            ok = (t.BlendEnable == 0 || t.BlendEnable == 1) && (t.LogicOpEnable == 0 || t.LogicOpEnable == 1) && t.SrcBlend >= 1 &&
                 t.SrcBlend <= 19 && t.DestBlend >= 1 && t.DestBlend <= 19 && t.BlendOp >= 1 && t.BlendOp <= 5 && t.SrcBlendAlpha >= 1 &&
                 t.SrcBlendAlpha <= 19 && t.DestBlendAlpha >= 1 && t.DestBlendAlpha <= 19 && t.BlendOpAlpha >= 1 && t.BlendOpAlpha <= 5 &&
                 static_cast<unsigned>(t.LogicOp) <= 15 && t.RenderTargetWriteMask <= 15;
        }
        if (ok) {
            *out = d;
            *where = at;
            return true;
        }
    }
    return false;
}

SRWLOCK draw_lock = SRWLOCK_INIT;
std::unordered_map<const Wrapper*, BoundPipeline*>& translation_by_pipeline() { static std::unordered_map<const Wrapper*, BoundPipeline*> m; return m; }
std::unordered_map<u64, Translation*>& translation_by_key() { static std::unordered_map<u64, Translation*> m; return m; }
LONG translated_draws, untranslated_draws, translation_ready, translation_missing, null_bindings;
LONG null_by_cause[x2w_local::null_cause_count];  // null_bindings split by cause; the parts add up to it

void count_null_binding(x2w_local::NullCause cause)
{
    if (cause == x2w_local::NullCause::none) return;
    InterlockedIncrement(&null_bindings);
    InterlockedIncrement(&null_by_cause[static_cast<int>(cause)]);
}
LONG frame_number;

u64 fnv_bytes(const void* data, std::size_t bytes, u64 hash = 1469598103934665603ull)
{
    for (std::size_t i = 0; i < bytes; ++i) hash = (hash ^ static_cast<const unsigned char*>(data)[i]) * 1099511628211ull;
    return hash;
}

bool cache_path(wchar_t* out, const wchar_t* prefix, u64 key, const wchar_t* extension)
{
    wchar_t dir[MAX_PATH];
    const DWORD n = GetEnvironmentVariableW(L"X2W_SHADER_CACHE", dir, MAX_PATH);
    if (!n || n >= MAX_PATH) return false;
    swprintf(out, MAX_PATH, L"%s\\%s%016llx%s", dir, prefix, key, extension);
    return true;
}

bool read_text(const wchar_t* path, std::string& out)
{
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    char buffer[8192];
    DWORD got = 0;
    while (ReadFile(file, buffer, sizeof(buffer), &got, nullptr) && got) out.append(buffer, got);
    CloseHandle(file);
    return true;
}

// Publish only complete inputs. The offline scanner treats .a and .rtv as ready markers.
bool publish_shader_request(u64 key, const wchar_t* ext, const void* data, std::size_t bytes)
{
    wchar_t path[MAX_PATH], temporary[MAX_PATH], suffix[32];
    if (!cache_path(path, L"req_", key, ext)) return false;
    if (GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES) return true;
    swprintf(suffix, std::size(suffix), L"%s.%lu.tmp", ext, GetCurrentProcessId());
    if (!cache_path(temporary, L"req_", key, suffix)) return false;
    HANDLE file = CreateFileW(temporary, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    DWORD written = 0;
    const bool complete = WriteFile(file, data, static_cast<DWORD>(bytes), &written, nullptr) && written == bytes;
    CloseHandle(file);
    if (complete && MoveFileExW(temporary, path, 0)) return true;
    DeleteFileW(temporary);
    return GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES;
}

void load_translation(Translation& tr, const std::string& rtv_request = {})
{
    tr.state = 2;
    wchar_t path[MAX_PATH];
    std::string manifest;
    if (!cache_path(path, L"", tr.key, L".txt")) return;
    if (!read_text(path, manifest)) {
        if (!tr.retry_manifest_at) {
            InterlockedIncrement(&translation_missing);
            static LONG missing;
            if (rtv_request.empty() && first_calls(missing, 12))
                log("x2w-draw no translation for pipeline %016llx", tr.key);
        }
        tr.retry_manifest_at = GetTickCount64() + 1000;
        return;
    }
    // The writer publishes this marker last. A malformed published result stays rejected.
    const bool previously_missing = tr.retry_manifest_at != 0;
    tr.retry_manifest_at = 0;
    bool matched_rtv = false;
    for (std::size_t at = 0; at < manifest.size();) {
        std::size_t end = manifest.find('\n', at);
        if (end == std::string::npos) end = manifest.size();
        std::string line = manifest.substr(at, end - at);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        at = end + 1;
        int i = 0, x = 0, y = 0;
        char kind[16] = "", source[16] = "";
        if (line.compare(0, 5, "rtv1 ") == 0) {
            if (matched_rtv || line != rtv_request) {
                log("x2w-draw RTV manifest mismatch for %016llx", tr.key);
                return;
            }
            matched_rtv = true;
        } else if (std::sscanf(line.c_str(), "udsrc %d %15s %d %d", &i, source, &x, &y) >= 2) {
            Translation::Ud ud{-1, x, y};
            if (!std::strcmp(source, "rect")) ud.kind = Translation::rect;
            else if (!std::strcmp(source, "basevertex")) ud.kind = Translation::base_vertex;
            else if (!std::strcmp(source, "startinstance")) ud.kind = Translation::start_instance;
            else if (!std::strcmp(source, "rootconst")) ud.kind = Translation::root_constant;
            else if (!std::strcmp(source, "stride")) ud.kind = Translation::stride;
            else if (!std::strcmp(source, "mrtbits") && !rtv_request.empty()) ud.kind = Translation::mrt_bits;
            if (static_cast<std::size_t>(i) == tr.uds.size() && ud.kind >= 0) tr.uds.push_back(ud);
        } else if (std::sscanf(line.c_str(), "srv %d %15s %15s %d %d", &i, kind, source, &x, &y) >= 4) {
            const int texture_kind = !std::strcmp(kind, "tex2d") ? 1 : !std::strcmp(kind, "tex2da") ? 2 :
                                     !std::strcmp(kind, "tex3d") ? 3 : !std::strcmp(kind, "tex1d") ? 4 : 0;
            Translation::Srv srv{texture_kind, !std::strcmp(source, "vb") ? 1 : !std::strcmp(source, "root") ? 2 : 0, x, y};
            if (static_cast<std::size_t>(i) == tr.srvs.size()) tr.srvs.push_back(srv);
        } else if (line.compare(0, 6, "wmask ") == 0) {
            if (!tr.write_masks.read_manifest_line(line)) {
                log("x2w-draw invalid write mask manifest line for %016llx: %s", tr.key, line.c_str());
                return;
            }
        } else if (!tr.samplers.read_manifest_line(line.c_str())) {
            log("x2w-draw invalid sampler heap manifest for %016llx", tr.key);
            return;
        }
    }
    if (matched_rtv != !rtv_request.empty()) {
        log("x2w-draw RTV manifest missing for %016llx", tr.key);
        return;
    }
    if (tr.uds.empty() || tr.uds.size() > 32 || tr.srvs.size() > 32) return;
    if (!cache_path(path, L"", tr.key, L".vs.cso") || FAILED(D3DReadFileToBlob(path, &tr.vs))) return;
    if (!cache_path(path, L"", tr.key, L".ps.cso") || FAILED(D3DReadFileToBlob(path, &tr.ps))) return;
    const HRESULT hr = x2w_startup::create_translated_root(device_of(the_device), static_cast<UINT>(tr.uds.size()),
        static_cast<UINT>(tr.srvs.size()), tr.samplers, &tr.root);
    if (FAILED(hr)) {
        log("x2w-draw root signature for %016llx failed 0x%08lx", tr.key, static_cast<unsigned long>(hr));
        return;
    }
    tr.state = 1;
    if (previously_missing) InterlockedDecrement(&translation_missing);
    InterlockedIncrement(&translation_ready);
    log("x2w-draw pipeline %016llx translated: %zu entry values, %zu resources, %zu static samplers, sampler_heap=%u",
        tr.key, tr.uds.size(), tr.srvs.size(), tr.samplers.statics.size(), tr.samplers.heap ? 1u : 0u);
}

// draw_lock held. Both pipeline forms use the same request writer and shader cache.
Translation* cached_translation(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes,
                                const std::vector<unsigned char>& root_blob, const std::string& layout = {})
{
    // Keep pre-fix desktop bytecode and manifests out of this ABI's cache.
    // tools/startup/gfx2hlsl.py hashes the same bytes, without a trailing NUL.
    constexpr char revision[] = "x2w-startup-shader-9";
    const u64 seed = fnv_bytes(revision, sizeof(revision) - 1);
    u64 key = fnv_bytes(root_blob.data(), root_blob.size(), fnv_bytes(b, b_bytes, fnv_bytes(a, a_bytes, seed)));
    // A stream's input layout (stream_input_layout) feeds the vertex fetch, so it is part of the key.
    if (!layout.empty()) key = fnv_bytes(layout.data(), layout.size(), key);
    auto& entry = translation_by_key()[key];
    if (!entry) {
        entry = new Translation;
        entry->key = key;
        // A legacy cache hit still needs source blobs for late RTV specialization.
        if (publish_shader_request(key, L".b", b, b_bytes) &&
            publish_shader_request(key, L".root", root_blob.data(), root_blob.size()) &&
            (layout.empty() || publish_shader_request(key, L".ia", layout.data(), layout.size())))
            publish_shader_request(key, L".a", a, a_bytes);
        load_translation(*entry);
    }
    return entry;
}

// The INPUT_LAYOUT subobject as the request text tools/startup/gfx2hlsl.py reads: "ia1 <count>", then one
// "<semantic> <index> <format> <slot> <offset> <class> <step>" line per element. False when it cannot be read.
bool stream_input_layout(const unsigned char* subobject, std::string& text)
{
    D3D12_INPUT_LAYOUT_DESC layout;
    std::memcpy(&layout, subobject + 8, sizeof(layout));
    if (!layout.NumElements) return true;
    if (layout.NumElements > D3D12_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT ||
        !x2w_local::readable(layout.pInputElementDescs, sizeof(D3D12_INPUT_ELEMENT_DESC) * layout.NumElements)) return false;
    text = "ia1 " + std::to_string(layout.NumElements) + "\n";
    std::string shown;
    for (UINT i = 0; i < layout.NumElements; ++i) {
        const D3D12_INPUT_ELEMENT_DESC& e = layout.pInputElementDescs[i];
        std::size_t length = 0;
        while (length < 64 && x2w_local::readable(e.SemanticName + length, 1) && e.SemanticName[length]) {
            const char c = e.SemanticName[length];
            if (!(std::isalpha(static_cast<unsigned char>(c)) || c == '_' || (length && std::isdigit(static_cast<unsigned char>(c))))) return false;
            ++length;
        }
        if (!length || length == 64) return false;
        char line[160];
        std::snprintf(line, sizeof(line), "%.*s %u %u %u %u %u %u\n", static_cast<int>(length), e.SemanticName, e.SemanticIndex,
                      static_cast<unsigned>(e.Format), e.InputSlot, e.AlignedByteOffset, static_cast<unsigned>(e.InputSlotClass),
                      e.InstanceDataStepRate);
        text += line;
        std::snprintf(line, sizeof(line), "%s%u/%u/%u/%u", i ? " " : "", static_cast<unsigned>(e.Format), e.InputSlot,
                      e.AlignedByteOffset, static_cast<unsigned>(e.InputSlotClass));
        shown += line;
    }
    static LONG logged;
    if (first_calls(logged, 24))
        log("x2w-draw stream input layout: %u elements (format/slot/offset/class) %s", layout.NumElements, shown.c_str());
    return true;
}

// Resolve the shader pointers and wrapped root while the caller still owns the stream.
Translation* stream_translation(const D3D12_PIPELINE_STATE_STREAM_DESC& desc, D3D12_BLEND_DESC& blend, bool& has_blend,
                                D3D12_DEPTH_STENCIL_DESC& depth_stencil, bool& has_depth_stencil, DXGI_FORMAT& dsv_format)
{
    const auto* stream = static_cast<const unsigned char*>(desc.pPipelineStateSubobjectStream);
    if (!desc.SizeInBytes || desc.SizeInBytes > 0x2000 || desc.SizeInBytes % 8 ||
        !x2w_local::readable(stream, desc.SizeInBytes)) return nullptr;

    const Wrapper* root = nullptr;
    D3D12_SHADER_BYTECODE stages[2]{};
    std::string layout;
    unsigned seen = 0;
    for (std::size_t at = 0; at < desc.SizeInBytes;) {
        unsigned type;
        std::memcpy(&type, stream + at, sizeof(type));
        if (type >= std::size(stream_sizes) || stream_sizes[type] > desc.SizeInBytes - at || (seen & (1u << type)))
            return nullptr;
        seen |= 1u << type;
        if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE) {
            std::memcpy(&root, stream + at + 8, sizeof(root));
        } else if (type >= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS && type <= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS) {
            D3D12_SHADER_BYTECODE shader;
            std::memcpy(&shader, stream + at + 8, sizeof(shader));
            if (shader.pShaderBytecode || shader.BytecodeLength) {
                if (type > D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS || shader.BytecodeLength < 32 ||
                    shader.BytecodeLength > (1u << 20) || !x2w_local::readable(shader.pShaderBytecode, shader.BytecodeLength) ||
                    std::memcmp(shader.pShaderBytecode, "DXBC", 4)) return nullptr;
                stages[type - D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS] = shader;
            }
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND) {
            std::memcpy(&blend, stream + at + sizeof(type), sizeof(blend));
            has_blend = true;
        } else if (type == 11 || type == 21) {  // DEPTH_STENCIL (56B) / DEPTH_STENCIL1 (64B)
            std::memcpy(&depth_stencil, stream + at + sizeof(type), sizeof(depth_stencil));
            has_depth_stencil = true;
            if (type == 21) {  // DEPTH_STENCIL1 appends DepthBoundsTestEnable; it has no translated equivalent
                BOOL bounds = FALSE;
                std::memcpy(&bounds, stream + at + sizeof(type) + sizeof(depth_stencil), sizeof(bounds));
                if (bounds) return nullptr;
            }
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT) {
            // Unreadable: translate without it; gfx2hlsl refuses a vertex shader that needs it.
            if (!stream_input_layout(stream + at, layout)) {
                layout.clear();
                log("x2w-draw stream input layout unreadable; requesting the translation without it");
            }
        } else if (type == 16) {  // DEPTH_STENCIL_FORMAT (8B)
            DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
            std::memcpy(&fmt, stream + at + sizeof(type), sizeof(fmt));
            dsv_format = desktop_format(fmt);
        }
        at += stream_sizes[type];
    }
    if (!root || !stages[0].pShaderBytecode || !stages[1].pShaderBytecode) return nullptr;
    std::vector<unsigned char> root_blob;
    AcquireSRWLockShared(&note_lock);
    const auto found = root_blobs().find(root);
    if (found != root_blobs().end()) root_blob = found->second;
    ReleaseSRWLockShared(&note_lock);
    if (root_blob.empty()) return nullptr;
    AcquireSRWLockExclusive(&draw_lock);
    auto* result = cached_translation(stages[0].pShaderBytecode, stages[0].BytecodeLength,
                                      stages[1].pShaderBytecode, stages[1].BytecodeLength, root_blob, layout);
    ReleaseSRWLockExclusive(&draw_lock);
    return result;
}

// draw_lock held.
BoundPipeline* translation_for(Wrapper* pipeline)
{
    auto& by_pipeline = translation_by_pipeline();
    const auto known = by_pipeline.find(pipeline);
    if (known != by_pipeline.end()) return known->second;
    auto* bound = new BoundPipeline;
    Translation*& result = bound->translation;
    const PipelineInfo* info = pipeline_info(pipeline);
    const PipelineInfo* own = info;
    {
        // The pipeline's own description first (a derived pipeline overrides state), then its base's state block.
        std::size_t where = 0;
        const char* from = "none";
        const char* refusal = nullptr;
        unsigned refused_targets = 0;
        for (int depth = 0; info && info->path == 55 && info->base && depth < 8; ++depth) {
            if (!bound->has_blend && x2w_derived_blend::last_blend(info->desc.data(), info->desc.size(), bound->blend, where)) {
                from = "derived blend patch";
                bound->has_blend = true;
            }
            info = pipeline_info(info->base);
        }
        if (!bound->has_blend && info && info->has_stream_blend) {
            from = "stream blend subobject";
            bound->blend = info->stream_blend;
        } else if (!bound->has_blend && info && info->path == 57 &&
                   x2w_startup::state_block_blend(info->state_a.data(), info->state_a.size(), false, &bound->blend, &refusal, &refused_targets)) {
            // The colour-blend register words of a slot 57 pipeline (state_block_blend.h, #1216).
            from = "colour-blend registers";
            where = x2w_startup::state_block_blend_offset;
        } else if (!bound->has_blend && info && find_blend(info->state_a, &bound->blend, &where)) from = "state block A";
        else if (!bound->has_blend && info && find_blend(info->state_b, &bound->blend, &where)) from = "state block B";
        bound->has_blend = from[0] != 'n';
        static LONG logged;
        if (first_calls(logged, 24)) {
            const auto& t = bound->blend.RenderTarget[0];
            log("x2w-draw pipeline %p (creation slot %d, base slot %d): blend from %s at +0x%zx: enable %d src %d dest %d op %d / alpha %d %d %d "
                "mask 0x%x; blocks %zu + %zu bytes", static_cast<void*>(pipeline), own ? own->path : 0, info ? info->path : 0, from, where,
                t.BlendEnable, t.SrcBlend, t.DestBlend, t.BlendOp, t.SrcBlendAlpha, t.DestBlendAlpha, t.BlendOpAlpha, t.RenderTargetWriteMask,
                info ? info->state_a.size() : 0, info ? info->state_b.size() : 0);
            if (refusal || refused_targets)
                log("x2w-draw   colour-blend registers not used: %s; refused targets 0x%x", refusal ? refusal : "refused", refused_targets);
            if (info && !bound->has_blend) {
                char hex[3 * 0x1a0 + 1];
                hex_text(info->state_a.data(), info->state_a.size() < 0x1a0 ? info->state_a.size() : 0x1a0, hex, sizeof(hex));
                log("x2w-draw   state A: %s", hex);
                hex_text(info->state_b.data(), info->state_b.size() < 0x1a0 ? info->state_b.size() : 0x1a0, hex, sizeof(hex));
                log("x2w-draw   state B: %s", hex);
                if (own != info && own) {
                    hex_text(own->desc.data(), own->desc.size() < 0x100 ? own->desc.size() : 0x100, hex, sizeof(hex));
                    log("x2w-draw   derived: %s", hex);
                }
            }
        }
    }
    if (info && info->path == 86) {
        result = info->stream_translation;
        if (info->has_stream_depth_stencil) {
            bound->depth_stencil = info->stream_depth_stencil;
            bound->has_depth_stencil = true;
        }
        bound->dsv_format = info->stream_dsv_format;
    }
    // X2W_DEPTH_STATE=1 (#1999, item A): the depth-stencil registers of a slot 57 pipeline's state block A. A slot 57
    // pipeline has no desktop DEPTH_STENCIL subobject, so has_depth_stencil can only be set here; the test keeps any
    // desktop state that was found first. A block the decoder refuses leaves the desktop default (depth and stencil off).
    if (info && info->path == 57 && !bound->has_depth_stencil && x2w_local::env_on("X2W_DEPTH_STATE")) {
        const auto chosen = x2w_pipeline_state::select_depth_stencil(true, info->state_a.data(), info->state_a.size());
        static LONG logged_applied, logged_refused;
        if (chosen.applied) {
            bound->depth_stencil = chosen.desc;
            bound->has_depth_stencil = bound->depth_from_registers = true;
            bound->has_stencil_ref = chosen.has_stencil_ref;
            bound->stencil_ref = chosen.stencil_ref;
        }
        if (first_calls(chosen.applied ? logged_applied : logged_refused, 24)) {
            if (chosen.applied)
                log("x2w-draw pipeline %p: depth-stencil from state block A: depth %d func %d write %d, stencil %d reference 0x%x%s%s",
                    static_cast<void*>(pipeline), chosen.desc.DepthEnable, chosen.desc.DepthFunc, chosen.desc.DepthWriteMask,
                    chosen.desc.StencilEnable, chosen.stencil_ref, chosen.depth_bounds ? "; depth bounds not applied" : "",
                    chosen.masks_differ ? "; back-face masks or reference differ, front used" : "");
            else
                log("x2w-draw pipeline %p: depth-stencil registers not used (%s); keeping the desktop state", static_cast<void*>(pipeline),
                    chosen.reason);
        }
    }
    if (info && info->path == 57 && info->desc.size() >= 0x60) {
        u64 words[12];
        std::memcpy(words, info->desc.data(), sizeof(words));
        Wrapper* root = as_wrapper(reinterpret_cast<const void*>(words[0]));
        std::vector<unsigned char> root_blob;
        if (root) {
            AcquireSRWLockShared(&note_lock);
            const auto found = root_blobs().find(root);
            if (found != root_blobs().end()) root_blob = found->second;
            ReleaseSRWLockShared(&note_lock);
        }
        const auto* a = reinterpret_cast<const void*>(words[5]);
        const auto* b = reinterpret_cast<const void*>(words[9]);
        const std::size_t a_bytes = static_cast<std::size_t>(words[6]), b_bytes = static_cast<std::size_t>(words[10]);
        if (!root_blob.empty() && a_bytes && b_bytes && a_bytes <= (1u << 20) && b_bytes <= (1u << 20) && x2w_local::readable(a, a_bytes) &&
            x2w_local::readable(b, b_bytes)) {
            result = cached_translation(a, a_bytes, b, b_bytes, root_blob);
        }
    }
    by_pipeline[pipeline] = bound;
    return bound;
}

// The request translation_for writes on a direct draw, for a pipeline that slot 61 refuses. Only direct draws poll
// for the manifest afterwards. Count first lookups separately from published request files.
LONG indirect_first_lookups;
void request_translation(Wrapper* pipeline)
{
    AcquireSRWLockExclusive(&draw_lock);
    const bool known = translation_by_pipeline().count(pipeline) != 0;
    translation_for(pipeline);
    ReleaseSRWLockExclusive(&draw_lock);
    if (!known) InterlockedIncrement(&indirect_first_lookups);
    static LONG lookups;
    if (first_calls(lookups, 16))
        log("x2w-render ExecuteIndirect looks up translation for pipeline %p (%s)", static_cast<void*>(pipeline),
            known ? "already known" : "new");
}

bool needs_rtv_variant(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R32G32B32A32_SINT:
    case DXGI_FORMAT_R32G32B32_UINT: case DXGI_FORMAT_R32G32B32_SINT:
    case DXGI_FORMAT_R16G16B16A16_UINT: case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R8G8B8A8_UINT: case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R16G16_UINT: case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R8G8_UINT: case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_UINT: case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_R8_UINT: case DXGI_FORMAT_R8_SINT:
        return true;
    default:
        return false;
    }
}

Translation* translation_for_targets(Translation* base, const Pass& pass)
{
    if (!base) return nullptr;
    if (std::none_of(pass.target_formats, pass.target_formats + pass.target_count, needs_rtv_variant)) {
        if (base->retry_manifest_at && GetTickCount64() >= base->retry_manifest_at)
            load_translation(*base);
        return base;
    }
    std::string request = "rtv1 " + std::to_string(base->key);
    for (UINT i = 0; i < pass.target_count; ++i)
        request += " " + std::to_string(static_cast<unsigned>(pass.target_formats[i]));
    const u64 key = fnv_bytes(request.data(), request.size());
    auto& entry = translation_by_key()[key];
    if (!entry) {
        entry = new Translation;
        entry->key = key;
        load_translation(*entry, request);
        if (entry->state != 1) {
            publish_shader_request(key, L".rtv", request.data(), request.size());
            log("x2w-draw no RTV translation for %016llx (%s)", key, request.c_str());
        }
    } else if (entry->retry_manifest_at && GetTickCount64() >= entry->retry_manifest_at) {
        load_translation(*entry, request);
    }
    return entry;
}

// The depth format a translated draw uses: the pass's own view, else the one the pipeline was created with.
inline DXGI_FORMAT pass_dsv_format(const BoundPipeline& bound, const Pass& pass)
{
    return pass.dsv_format != DXGI_FORMAT_UNKNOWN ? pass.dsv_format : bound.dsv_format;
}

ID3D12PipelineState* pipeline_for(BoundPipeline& bound, Translation& tr, const Pass& pass)
{
    const DXGI_FORMAT dsv_format = pass_dsv_format(bound, pass);
    u64 key = pass.target_count;
    for (UINT i = 0; i < pass.target_count; ++i) key = key * 131 + static_cast<u64>(pass.target_formats[i]);
    key = fnv_bytes(&dsv_format, sizeof(dsv_format), key);
    // A state decoded from registers needs a depth view to act on; with none bound the pass keeps the desktop default.
    const bool use_depth_stencil = bound.has_depth_stencil && (!bound.depth_from_registers || dsv_format != DXGI_FORMAT_UNKNOWN);
    if (use_depth_stencil) key = fnv_bytes(&bound.depth_stencil, sizeof(bound.depth_stencil), key);
    const auto found = bound.pipelines.find(key);
    if (found != bound.pipelines.end()) return found->second;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = tr.root;
    desc.VS = {tr.vs->GetBufferPointer(), tr.vs->GetBufferSize()};
    desc.PS = {tr.ps->GetBufferPointer(), tr.ps->GetBufferSize()};
    // Blend state as found in the pipeline's state block, else opaque. Depth-stencil state and
    // DSVFormat come from the stream PSO (or bound DSV), else disabled. Raster state: no culling.
    for (auto& target : desc.BlendState.RenderTarget)
        target = {FALSE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_BLEND_ONE, D3D12_BLEND_ZERO,
                  D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
    if (bound.has_blend) desc.BlendState = bound.blend;
    // Channels the translated exports leave unwritten stay out of the write mask (#1674).
    if (!x2w_startup::apply_export_write_masks(desc.BlendState, tr.write_masks, pass.target_count)) {
        log("x2w-draw export write masks incompatible with shared logic op for %016llx", tr.key);
        bound.pipelines[key] = nullptr;
        return nullptr;
    }
    if (use_depth_stencil) desc.DepthStencilState = bound.depth_stencil;
    desc.SampleMask = ~0u;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = pass.target_count;
    for (UINT i = 0; i < pass.target_count; ++i) desc.RTVFormats[i] = pass.target_formats[i];
    desc.DSVFormat = dsv_format;
    desc.SampleDesc.Count = 1;
    ID3D12PipelineState* pipeline = nullptr;
    const HRESULT hr = device_of(the_device)->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pipeline));
    log("x2w-draw desktop pipeline for %016llx, %u targets (format %d) -> 0x%08lx (dsv %d, depth %d)", tr.key,
        pass.target_count, pass.target_count ? pass.target_formats[0] : 0, static_cast<unsigned long>(hr),
        static_cast<int>(dsv_format), desc.DepthStencilState.DepthEnable ? 1 : 0);
    check_device("translated CreateGraphicsPipelineState", 0);
    bound.pipelines[key] = pipeline;
    return pipeline;
}

// Private shader-visible heap for the translated draws' tables, used as a ring.
constexpr UINT draw_heap_size = 1u << 18;
ID3D12DescriptorHeap* draw_heap;
Wrapper* draw_heap_device;
UINT draw_heap_increment;
volatile LONG draw_heap_cursor;

bool ensure_draw_heap(Wrapper* device)
{
    // The translated pipeline and upload ring are scoped to the current
    // startup render device. A list from another device cannot use this heap.
    if (!device || device != the_device) return false;
    if (draw_heap) return device == draw_heap_device;
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, draw_heap_size, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (FAILED(device_of(the_device)->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&draw_heap)))) return false;
    draw_heap_device = device;
    draw_heap_increment = device_of(the_device)->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

// A recorded table is immutable. Keep it alive until its submission completes,
// including when the title resets the command list with a different allocator.
using DrawSamplerHeap = x2w_startup::ComPtr<ID3D12DescriptorHeap>;
std::unordered_map<ID3D12CommandList*, std::vector<DrawSamplerHeap>> recorded_draw_sampler_heaps;
struct SubmittedDrawSamplerHeaps {
    x2w_startup::ComPtr<ID3D12Fence> fence;
    std::vector<DrawSamplerHeap> heaps;
};
std::vector<SubmittedDrawSamplerHeaps> submitted_draw_sampler_heaps;

void discard_draw_sampler_heaps(ID3D12CommandList* list)
{
    AcquireSRWLockExclusive(&draw_lock);
    recorded_draw_sampler_heaps.erase(list);
    ReleaseSRWLockExclusive(&draw_lock);
}

// draw_lock held. The lists have already been executed on this queue. A closed list can be
// submitted again until its Reset, so each submit takes its own references and the recording
// keeps its own until discard_draw_sampler_heaps.
void retire_draw_sampler_heaps(ID3D12CommandQueue* queue, const std::vector<ID3D12CommandList*>& lists)
{
    submitted_draw_sampler_heaps.erase(std::remove_if(submitted_draw_sampler_heaps.begin(),
        submitted_draw_sampler_heaps.end(), [](const SubmittedDrawSamplerHeaps& batch) {
            return batch.fence && batch.fence->GetCompletedValue() != 0;
        }), submitted_draw_sampler_heaps.end());
    SubmittedDrawSamplerHeaps batch;
    for (auto* list : lists) {
        auto it = recorded_draw_sampler_heaps.find(list);
        if (it == recorded_draw_sampler_heaps.end()) continue;
        for (const auto& heap : it->second) batch.heaps.push_back(heap);
    }
    if (batch.heaps.empty()) return;
    x2w_startup::ComPtr<ID3D12Device> device;
    HRESULT hr = queue->GetDevice(IID_PPV_ARGS(&device));
    if (SUCCEEDED(hr)) hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&batch.fence));
    if (SUCCEEDED(hr)) hr = queue->Signal(batch.fence.Get(), 1);
    if (FAILED(hr)) {
        // Without a completion point the submitted descriptors still belong
        // to the queue. Retain them instead of reclaiming them early.
        batch.fence.Reset();
        log("x2w-draw sampler heap retirement fence failed 0x%08lx", static_cast<unsigned long>(hr));
    }
    submitted_draw_sampler_heaps.push_back(std::move(batch));
}

bool resolve_draw_sampler(const x2w_startup::SamplerWords& words, D3D12_SAMPLER_DESC& sampler)
{
    if (words[0] != view_token_magic) {
        sampler = x2w_startup::dynamic_sampler(words);
        return true;
    }
    AcquireSRWLockShared(&note_lock);
    const auto& records = view_records();
    const bool valid = words[2] == 22 && words[1] < records.size() && records[words[1]].slot == 22 &&
                       records[words[1]].desc_bytes == sizeof(sampler);
    if (valid) std::memcpy(&sampler, records[words[1]].desc, sizeof(sampler));
    ReleaseSRWLockShared(&note_lock);
    return valid;
}

// draw_lock held.
bool prepare_draw_sampler_heap(Pass& pass, ID3D12GraphicsCommandList* list, bool reserved_fallback)
{
    auto snapshot = x2w_startup::snapshot_samplers(pass.title_samplers, reserved_fallback);
    if (pass.sampler_snapshot && *pass.sampler_snapshot == snapshot) return true;
    x2w_startup::NullSamplerCounts counts;
    DrawSamplerHeap heap;
    const HRESULT hr = x2w_startup::create_sampler_snapshot_heap(device_of(the_device), snapshot,
        resolve_draw_sampler, heap.GetAddressOf(), counts);
    if (FAILED(hr)) {
        log("x2w-draw sampler heap creation failed 0x%08lx", static_cast<unsigned long>(hr));
        return false;
    }
    static LONG reports;
    if (first_calls(reports, 12))
        log("x2w-draw sampler heap: null_sampler_unbound=%u null_sampler_out_of_range=%u "
            "null_sampler_unreadable=%u null_sampler_invalid_token=%u",
            counts.unbound, counts.out_of_range, counts.unreadable, counts.invalid_token);
    recorded_draw_sampler_heaps[list].push_back(heap);
    pass.sampler_heap = std::move(heap);
    pass.sampler_snapshot = std::make_unique<x2w_startup::SamplerSnapshot>(std::move(snapshot));
    return true;
}

// Copies of title memory made for the current frame, found by (address, bytes). A title may rewrite the same
// address between two draws of one frame (a dynamic constant buffer, a reused vertex or index range), so an
// entry keeps the bytes it was made from and is reused only while the title's memory still matches them. The
// upload ring is write-combined memory that is slow to read, so the comparison is against this copy, not the ring.
struct FrameCopy {
    u64 address = 0;
    u64 offset = 0;
    std::vector<unsigned char> content;
};
std::unordered_map<u64, FrameCopy>& frame_copies() { static std::unordered_map<u64, FrameCopy> m; return m; }
LONG frame_copies_frame = -1;

// Callers have just probed that [address, address + bytes) is readable (readable_window), and draw_lock is held.
bool frame_copy(u64 address, std::size_t bytes, u64* offset)
{
    if (frame_copies_frame != frame_number) {
        frame_copies().clear();
        frame_copies_frame = frame_number;
    }
    const u64 key = address * 1099511628211ull ^ bytes;
    const auto* title = reinterpret_cast<const unsigned char*>(address);
    const auto found = frame_copies().find(key);
    const bool known = found != frame_copies().end() && found->second.address == address && found->second.content.size() == bytes;
    if (known && std::memcmp(found->second.content.data(), title, bytes) == 0) {
        *offset = found->second.offset;
        return true;
    }
    u64 fresh = 0;
    auto* ring = reserve_upload_ring(bytes, &fresh);
    if (!ring) return false;
    // Cache insertion and snapshot allocation are optional. Fill the ring directly if they fail.
    const unsigned char* source = title;
    try {
        std::vector<unsigned char> content(title, title + bytes);
        // Use the stored snapshot to fill the ring so the GPU bytes and subsequent comparisons agree.
        auto entry = frame_copies().insert_or_assign(key, FrameCopy{address, fresh, std::move(content)});
        source = entry.first->second.content.data();
    } catch (const std::bad_alloc&) {
        // The ring slice is already reserved. The draw can proceed without another CPU allocation.
    }
    std::memcpy(ring, source, bytes);
    if (known) {
        static LONG refreshed;
        if (first_calls(refreshed, 4))
            log("x2w-draw frame copy of 0x%llx (%zu bytes) refreshed: the title changed it since an earlier draw of this frame", address, bytes);
    }
    *offset = fresh;
    return true;
}

void buffer_view(D3D12_CPU_DESCRIPTOR_HANDLE dest, ID3D12Resource* buffer, u64 offset, std::size_t bytes)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_R32_TYPELESS;
    desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.Buffer.FirstElement = buffer ? offset / 4 : 0;
    desc.Buffer.NumElements = buffer ? static_cast<UINT>((bytes + 3) / 4) : 1;
    desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    device_of(the_device)->CreateShaderResourceView(buffer, &desc, dest);
}

// Texture contents. Linear backing uses desktop copyable footprints; an announced BC7 4 KiB
// surface must be detiled into the footprint first. A texture is uploaded on its first bind and
// again when its memory changed: XG copies announce themselves (x2w_d3d12_texture_written), and
// CPU-backed textures up to 32 MiB are also re-hashed for writes the title makes on its own.
// Recheck single-mip R8 and A8 planes on their first bind each frame and other textures every 15 frames.
// Targets and UAVs are GPU-written and never uploaded.
struct TextureUpload {
    u64 hash = 0, generation = ~0ull;
    LONG checked_frame = -1000;
    bool done = false, tiled = false;
};
std::unordered_map<ID3D12Resource*, TextureUpload>& texture_uploads() { static std::unordered_map<ID3D12Resource*, TextureUpload> m; return m; }
LONG textures_uploaded, texture_bytes_uploaded, textures_unreadable;
u64 hash_memory(const unsigned char* data, std::size_t bytes)
{
    u64 h = 0xcbf29ce484222325ull;
    const std::size_t words = bytes / 8;
    const u64* w = reinterpret_cast<const u64*>(data);
    for (std::size_t i = 0; i < words; ++i) h = (h ^ w[i]) * 0x100000001b3ull;
    for (std::size_t i = words * 8; i < bytes; ++i) h = (h ^ data[i]) * 0x100000001b3ull;
    return h;
}
void upload_texture(Wrapper* self, Wrapper* resource, ID3D12Resource* texture)
{
    auto* const memory = static_cast<const unsigned char*>(resource->own().title_memory);
    auto* list = recording(self);
    const D3D12_RESOURCE_DESC desc = texture->GetDesc();
    // An address-only single-element texture (2D or 3D) uploads its one texel through the footprint path.
    const bool single = resource->own().tiled_address_srv && x2w_startup::single_element_bytes(desc) != 0;
    const bool uploadable = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D || single ||
        (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && desc.DepthOrArraySize == 1 && desc.Height == 1);
    {
        static LONG reasons;
        if ((!memory || (desc.Flags & 0x7) || !uploadable) && !texture_uploads().count(texture) &&
            first_calls(reasons, 12)) {
            texture_uploads()[texture].done = true;
            log("x2w-draw texture not uploaded %p: %llux%u format %d flags 0x%x dim %d title memory %p", static_cast<void*>(texture),
                desc.Width, desc.Height, desc.Format, desc.Flags, desc.Dimension, static_cast<const void*>(memory));
        }
    }
    if (!memory || !list) return;
    if (desc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) return;
    if (!uploadable) return;
    const UINT mips = desc.MipLevels ? desc.MipLevels : 1;
    const UINT slices = desc.DepthOrArraySize ? desc.DepthOrArraySize : 1;
    if (mips > 15) return;
    const x2w_startup::SliceFootprints plan(device_of(the_device), desc, mips);
    const UINT64 slice_total = plan.size;
    if (slice_total == ~0ull || !slice_total) return;
    const u64 stride = (slice_total + 0xFFFF) & ~u64{0xFFFF};
    TextureUpload& state = texture_uploads()[texture];
    TextureWrite written;
    AcquireSRWLockShared(&written_lock);
    if (const auto found = written_textures().find(reinterpret_cast<u64>(memory)); found != written_textures().end())
        written = found->second;
    ReleaseSRWLockShared(&written_lock);
    const auto* original = resource_info(resource);
    const bool tiled_chain = original && x2w_startup::bc64::supports(original->console);
    const bool tiled_address = resource->own().tiled_address_srv;
    const bool tiled_bc7 = !tiled_chain && !tiled_address && written.bc7_width != 0;
    const bool tiled = tiled_bc7 || tiled_address || tiled_chain;
    const auto generation = written.generation;
    const x2w_startup::Bc16Surface surface(tiled_bc7 ? written.bc7_width : static_cast<UINT>(desc.Width),
                                         tiled_bc7 ? written.bc7_height : desc.Height);
    if (tiled_bc7) {
        if (!original || !x2w_startup::supports_bc7_4k_s(original->console) ||
            original->console.Width != written.bc7_width || original->console.Height != written.bc7_height ||
            desc.Format != DXGI_FORMAT_BC7_UNORM || desc.Width != surface.columns * surface.block_texels ||
            desc.Height != surface.rows * surface.block_texels) return;
    }
    if (tiled && !tiled_chain && !single && (plan.count != 1 || plan.rows[0] != surface.rows ||
                  plan.row_bytes[0] != u64{surface.columns} * surface.block_bytes)) return;
    const u64 total = tiled_chain ? x2w_startup::bc64::Chain(original->console).bytes() : tiled && !single ? surface.size() : stride * (slices - 1) + slice_total;
    const bool announced = state.generation != generation;
    const UINT elapsed = static_cast<UINT>(frame_number) - static_cast<UINT>(state.checked_frame);
    // A single texel (at most 16 bytes) is rehashed on every bind, so a write between two draws is seen.
    const bool recheck = single || ((tiled_address || tiled_chain) ? elapsed != 0
        : x2w_startup::recheck_cpu_texture(desc, total, elapsed));
    if (state.done && !announced && !recheck) return;
    if (readable_window(reinterpret_cast<u64>(memory), static_cast<std::size_t>(total)) < total) {
        if (!state.done) InterlockedIncrement(&textures_unreadable);
        state.done = true;
        state.generation = generation;
        return;
    }
    state.checked_frame = frame_number;
    const u64 hash = hash_memory(memory, static_cast<std::size_t>(total));
    if (state.done && hash == state.hash && state.tiled == tiled) {
        state.generation = generation;
        return;
    }
    require_state(self, texture, D3D12_RESOURCE_STATE_COPY_DEST);
    if (tiled_chain) {
        const x2w_startup::bc64::Chain chain(original->console);
        for (UINT slice = 0; slice < slices; ++slice) {
            for (UINT i = 0; i < mips; ++i) {
                const u64 bytes = plan.bytes(i);
                u64 offset = 0;
                auto* rows = reserve_upload_ring(static_cast<std::size_t>(bytes), &offset);
                if (!rows) {
                    state.done = false;
                    return;
                }
                std::memset(rows, 0, static_cast<std::size_t>(bytes));
                chain.copy<false>(i, slice, rows, memory, plan.placed[i].Footprint.RowPitch);
                plan.upload(list, texture, upload_ring, mips, slices, slice, i, offset);
            }
        }
    } else if (tiled && !single) {
        u64 offset = 0;
        const auto bytes = plan.bytes(0);
        auto* rows = reserve_upload_ring(static_cast<std::size_t>(bytes), &offset);
        if (!rows) {
            state.done = false;
            return;
        }
        std::memset(rows, 0, static_cast<std::size_t>(bytes));
        surface.detile(memory, rows, plan.placed[0].Footprint.RowPitch);
        plan.upload(list, texture, upload_ring, 1, 1, 0, 0, offset);
    } else {
        for (UINT s = 0; s < slices; ++s)
            for (UINT i = 0; i < plan.count; ++i) {
                const auto& fp = plan.placed[i];
                const u64 bytes = plan.bytes(i);
                u64 offset = 0;
                if (!ring_copy(reinterpret_cast<u64>(memory) + stride * s + fp.Offset, static_cast<std::size_t>(bytes), &offset)) {
                    state.done = false;
                    return;
                }
                plan.upload(list, texture, upload_ring, mips, slices, s, i, offset);
            }
    }
    state.hash = hash;
    state.generation = generation;
    state.tiled = tiled;
    state.done = true;
    InterlockedIncrement(&textures_uploaded);
    InterlockedExchangeAdd(&texture_bytes_uploaded, static_cast<LONG>(total >> 10));
    static LONG shown;
    if (first_calls(shown, 12))
        log("x2w-draw texture upload %p %llux%u x%u mips %u format %d from %p (%llu KiB)%s", static_cast<void*>(texture), desc.Width,
            desc.Height, slices, mips, desc.Format, static_cast<const void*>(memory), total >> 10, announced ? " (announced write)" : "");
}

// Per-draw dumps (X2W_DRAW_DUMP_FRAME=<frame>, X2W_DRAW_DUMP_DIR=<folder>): during that frame, render
// target 0 is copied to a readback buffer after each translated draw (at most 48).
// Poll the fence of the queue that actually executed the recording before writing its image.
using x2w_startup::DrawDump;
LONG draw_dump_frame()
{
    static const LONG frame = [] {
        char value[16]{};
        return GetEnvironmentVariableA("X2W_DRAW_DUMP_FRAME", value, sizeof(value)) ? std::atol(value) : 0L;
    }();
    return frame;
}
void draw_dump_after(Wrapper* self, ID3D12GraphicsCommandList* list, u64 key, UINT count)
{
    Pass& pass = pass_of(self);
    const LONG wanted = draw_dump_frame();
    if (!wanted || frame_number < wanted || frame_number > wanted + 1 || !pass.target0 || draw_dumps().full()) return;
    const D3D12_RESOURCE_DESC desc = pass.target0->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
        !x2w_draw_bmp::supported(desc.Format)) return;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    device_of(the_device)->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes, &total);
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* readback = nullptr;
    if (FAILED(device_of(the_device)->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                              IID_PPV_ARGS(&readback)))) return;
    DrawDump dump{};
    dump.buffer.Attach(readback);
    dump.footprint = footprint;
    dump.rows = rows;
    dump.key = key;
    dump.count = count;
    dump.format = desc.Format;
    if (!draw_dumps().record(list, std::move(dump))) return;
    require_state(self, pass.target0, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION target{readback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    target.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION source{pass.target0, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    source.SubresourceIndex = 0;
    list->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
    require_state(self, pass.target0, D3D12_RESOURCE_STATE_RENDER_TARGET);
}
void draw_dump_flush()
{
    if (!draw_dump_frame()) return;
    auto completed = draw_dumps().take_completed();
    if (completed.empty()) return;
    char folder[MAX_PATH] = ".";
    GetEnvironmentVariableA("X2W_DRAW_DUMP_DIR", folder, sizeof(folder));
    int written = 0;
    for (const DrawDump& dump : completed) {
        const int index = static_cast<int>(dump.index);
        void* mapped = nullptr;
        if (FAILED(dump.buffer->Map(0, nullptr, &mapped))) continue;
        char path[MAX_PATH + 64];
        std::snprintf(path, sizeof(path), "%s\\draw_%02d_%016llx_%u.bmp", folder, index, dump.key, dump.count);
        if (FILE* file = std::fopen(path, "wb")) {
            const auto& fp = dump.footprint.Footprint;
            const auto* pixels = static_cast<const unsigned char*>(mapped) + dump.footprint.Offset;
            const bool complete = x2w_draw_bmp::write(file, pixels, fp.Width, dump.rows, fp.RowPitch, dump.format);
            if (std::fclose(file) == 0 && complete) ++written;
            else std::remove(path);
        }
        const D3D12_RANGE no_writes{};
        dump.buffer->Unmap(0, &no_writes);
    }
    log("x2w-draw per-draw dump: %d of %zu completed target copies written to %s", written, completed.size(), folder);
}

bool draw_translated(Wrapper* self, bool indexed, UINT count, UINT instances, UINT start, INT base_vertex, UINT start_instance)
{
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    const bool rect = pass.topology == 0x11;
    if (!list || !pass.pipeline || !pass.targets_bound || !(rect || pass.topology == 4 || pass.topology == 5) || (rect && indexed)) return false;
    AcquireSRWLockExclusive(&draw_lock);
    BoundPipeline* bound = translation_for(pass.pipeline);
    Translation* tr = bound ? translation_for_targets(bound->translation, pass) : nullptr;
    ID3D12PipelineState* pipeline = tr && tr->state == 1 && ensure_draw_heap(self->own().list_device) ? pipeline_for(*bound, *tr, pass) : nullptr;
    if (!pipeline || (tr->samplers.heap && !prepare_draw_sampler_heap(pass, list, tr->samplers.reserved_fallback))) {
        ReleaseSRWLockExclusive(&draw_lock);
        return false;
    }
    const UINT n = static_cast<UINT>(tr->srvs.size());
    UINT first = static_cast<UINT>(InterlockedExchangeAdd(&draw_heap_cursor, static_cast<LONG>(n))) % draw_heap_size;
    if (first + n > draw_heap_size) {
        first = 0;
        InterlockedExchange(&draw_heap_cursor, static_cast<LONG>(n));
    }
    std::uint32_t strides[32] = {};
    // Only the vertices this draw can reference are copied: [vertex_first, vertex_first + vertex_count).
    // The shader adds the "base vertex" entry value to the vertex index, so that value is biased by
    // -vertex_first and each stream's view starts at vertex_first.
    const UINT index_bytes = pass.ib.Format == DXGI_FORMAT_R16_UINT ? 2 : 4;
    const u64 index_address = pass.ib.BufferLocation + u64{start} * index_bytes;
    const std::size_t index_total = std::size_t{count} * index_bytes;
    const bool indices_ok = indexed && pass.ib.BufferLocation && readable_window(index_address, index_total) >= index_total;
    INT vertex_first = indexed ? 0 : static_cast<INT>(start);
    UINT vertex_count = indexed ? 0 : count;
    if (indices_ok) {
        std::uint32_t low = ~0u, high = 0;
        for (UINT k = 0; k < count; ++k) {
            const std::uint32_t value = index_bytes == 2 ? reinterpret_cast<const std::uint16_t*>(index_address)[k]
                                                         : reinterpret_cast<const std::uint32_t*>(index_address)[k];
            if (value < low) low = value;
            if (value > high) high = value;
        }
        if (low <= high) {
            vertex_first = base_vertex + static_cast<INT>(low);
            vertex_count = high - low + 1;
        }
    }
    if (vertex_first < 0) vertex_first = 0;
    for (UINT i = 0; i < n; ++i) {
        const Translation::Srv& srv = tr->srvs[i];
        const D3D12_CPU_DESCRIPTOR_HANDLE dest{draw_heap->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T{first + i} * draw_heap_increment};
        u64 address = 0, offset = 0;
        std::size_t bytes = 0;
        PlacedVertexView placed{};
        if (srv.source == 1 && srv.a >= 0 && srv.a < 8) {
            placed = pass.placed_vb[srv.a];
            address = pass.vb[srv.a].BufferLocation;
            bytes = pass.vb[srv.a].SizeInBytes;
            strides[i] = pass.vb[srv.a].StrideInBytes;
            const u64 skip = u64{static_cast<UINT>(vertex_first)} * strides[i];
            if (strides[i] && vertex_count && skip < bytes) {
                const u64 wanted = u64{vertex_count} * strides[i];
                address += skip;
                bytes = static_cast<std::size_t>(wanted < bytes - skip ? wanted : bytes - skip);
            }
        } else if (srv.source == 2 && srv.a < 16) {
            address = pass.root_address[srv.a];
            bytes = readable_window(address, 0x10000);
        }
        const u64 entry = srv.source == 0 && srv.a >= 0 && srv.a < 16 && srv.b >= 0 ?
            pass.view_tables[srv.a].record(static_cast<u64>(srv.b), 32) : 0;
        std::uint32_t words[8] = {};
        const bool entry_ok = entry && x2w_local::readable(reinterpret_cast<const void*>(entry), 32);
        if (entry_ok) std::memcpy(words, reinterpret_cast<const void*>(entry), 32);
        bool buffer_token_known = false;
        if (srv.kind >= 1 && srv.kind <= 4) {
            const auto dimension = srv.kind == 1 ? D3D12_SRV_DIMENSION_TEXTURE2D :
                                   srv.kind == 2 ? D3D12_SRV_DIMENSION_TEXTURE2DARRAY :
                                   srv.kind == 4 ? D3D12_SRV_DIMENSION_TEXTURE1D : D3D12_SRV_DIMENSION_TEXTURE3D;
            // A texture: the title's descriptor memory carries the id of the view it created there.
            ViewRecord record{};
            bool known = false;
            if (words[0] == view_token_magic) {
                AcquireSRWLockShared(&note_lock);
                known = words[1] < view_records().size();
                if (known) record = view_records()[words[1]];
                ReleaseSRWLockShared(&note_lock);
            }
            Wrapper* resource = known ? as_wrapper(reinterpret_cast<const void*>(record.resource)) : nullptr;
            D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.ViewDimension = dimension;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc.Texture2D.MipLevels = 1;  // shared MostDetailedMip and MipLevels in all supported texture views
            if (srv.kind == 2) desc.Texture2DArray.ArraySize = 1;
            ID3D12Resource* texture = sampled_address_texture(resource);
            if (texture && record.desc_bytes >= 40) {
                std::memcpy(&desc, record.desc, 40);
                desc.Format = desktop_format(desc.Format);
            }
            if (texture && x2w_startup::packed_target_format(desc.Format)) {
                // The view recorded from the title names a packed format: follow a stand-in texture,
                // and bind nothing for a texture the view does not fit (packed_target_format.h).
                const auto verdict = x2w_startup::packed_view(desc.Format, texture->GetDesc().Format, x2w_startup::packed_stand_in_of(texture));
                if (verdict == x2w_startup::PackedView::stand_in) desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                if (verdict == x2w_startup::PackedView::refused) {
                    static LONG refused;
                    if (first_calls(refused, 6))
                        log("x2w-draw   %016llx srv %u: packed format %d on a texture of format %d that does not stand in for it; binding null",
                            tr->key, i, desc.Format, texture->GetDesc().Format);
                    texture = nullptr;
                    desc = {};
                    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                    desc.ViewDimension = dimension;
                    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    desc.Texture2D.MipLevels = 1;
                    if (srv.kind == 2) desc.Texture2DArray.ArraySize = 1;
                }
            }
            if (texture) upload_texture(self, resource, texture);
            if (texture) require_state(self, texture, sampled_state(self, texture));
            const bool had_texture = texture != nullptr;
            bool mismatched = false;
            static LONG shown;
            if (first_calls(shown, 6))
                log("x2w-draw   %016llx srv %u texture: table entry 0x%llx words %08x %08x -> view %s, resource %p, desc %u bytes format %d",
                    tr->key, i, entry, words[0], words[1], known ? "known" : "UNKNOWN", static_cast<void*>(texture), record.desc_bytes, desc.Format);
            const bool wrong_1d_resource = texture && srv.kind == 4 &&
                (texture->GetDesc().Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D || texture->GetDesc().DepthOrArraySize != 1);
            if (texture && ((record.desc_bytes >= 40 && desc.ViewDimension != dimension) || wrong_1d_resource)) {
                static LONG mismatch;
                if (first_calls(mismatch, 6)) log("x2w-draw   %016llx srv %u: view dimension %d does not match the shader; binding null", tr->key, i, desc.ViewDimension);
                texture = nullptr;
                mismatched = true;
                desc.ViewDimension = dimension;
                desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                desc.Texture2DArray = {};
                desc.Texture2D.MipLevels = 1;
                if (srv.kind == 2) desc.Texture2DArray.ArraySize = 1;
            }
            x2w_local::BindingFacts facts;
            facts.texture = facts.from_table = true;
            facts.entry_readable = entry_ok;
            facts.token_magic = words[0] == view_token_magic;
            facts.token_known = known;
            facts.resource_wrapper = had_texture;
            facts.dimension_matches = !mismatched;
            count_null_binding(x2w_local::classify_null_binding(facts));
            device_of(the_device)->CreateShaderResourceView(texture, texture && record.desc_bytes < 40 ? nullptr : &desc, dest);
            continue;
        }
        if (srv.source == 0 && words[0] == view_token_magic && entry) {
            // A view the title created: a constant buffer view names its memory directly; a buffer
            // shader-resource view names a range of its resource's title memory.
            ViewRecord record{};
            AcquireSRWLockShared(&note_lock);
            buffer_token_known = words[1] < view_records().size();
            if (buffer_token_known) record = view_records()[words[1]];
            ReleaseSRWLockShared(&note_lock);
            if (record.slot == 17 && record.desc_bytes >= 12) {
                std::uint32_t size = 0;
                std::memcpy(&address, record.desc, 8);
                std::memcpy(&size, record.desc + 8, 4);
                bytes = size;
            } else if (record.slot == 18 && record.desc_bytes >= 40) {
                D3D12_SHADER_RESOURCE_VIEW_DESC view;
                std::memcpy(&view, record.desc, 40);
                Wrapper* resource = as_wrapper(reinterpret_cast<const void*>(record.resource));
                if (view.ViewDimension == D3D12_SRV_DIMENSION_BUFFER && resource && resource->own().title_memory) {
                    const bool typed_view = !view.Buffer.StructureByteStride && view.Format != DXGI_FORMAT_R32_TYPELESS;
                    const UINT element = view.Buffer.StructureByteStride ? view.Buffer.StructureByteStride
                                         : typed_view ? x2w_startup::typed_buffer_format_bytes(view.Format) : 4;
                    address = reinterpret_cast<u64>(resource->own().title_memory) + view.Buffer.FirstElement * element;
                    bytes = std::size_t{view.Buffer.NumElements} * element;
                    strides[i] = typed_view ? element : view.Buffer.StructureByteStride;
                }
            }
        }
        if (srv.source == 0 && words[0] != view_token_magic && entry) {
            // A hardware buffer descriptor the title wrote itself: base address, stride and record count.
            address = words[0] | (static_cast<u64>(words[1] & 0xffff) << 32);
            strides[i] = (words[1] >> 16) & 0x3fff;
            bytes = static_cast<std::size_t>(strides[i] ? strides[i] : 1) * words[2];
        }
        if (bytes > (16u << 20)) bytes = 16u << 20;
        // A buffer binding that ends up without a readable range counts as a null binding with its cause.
        const auto count_buffer = [&](bool bound) {
            x2w_local::BindingFacts buffer_facts;
            buffer_facts.from_table = srv.source == 0;
            buffer_facts.entry_readable = entry_ok;
            buffer_facts.token_magic = entry_ok && words[0] == view_token_magic;
            buffer_facts.token_known = buffer_token_known;
            buffer_facts.bound = bound;
            count_null_binding(x2w_local::classify_null_binding(buffer_facts));
        };
        if (placed.reject || (placed.owner && !InterlockedCompareExchange(&placed.owner->own().refs, 0, 0))) {
            buffer_view(dest, nullptr, 0, 0);
            count_buffer(false);
            continue;
        }
        if (placed.owner && bytes % 4 != 0) {
            // Preserve tight byte-sized streams. The desktop raw view is rounded
            // to a DWORD, so initialise its final padding without reading the guest.
            const auto padded = (bytes + 3) & ~std::size_t{3};
            auto* target = readable_window(address, bytes) >= bytes ? reserve_upload_ring(padded, &offset) : nullptr;
            if (target) {
                std::memcpy(target, reinterpret_cast<const void*>(address), bytes);
                std::memset(target + bytes, 0, padded - bytes);
            }
            buffer_view(dest, target ? upload_ring : nullptr, offset, padded);
            count_buffer(target != nullptr);
            continue;
        }
        if (placed.alias && address % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT == 0 && bytes) {
            require_state(self, placed.alias, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            buffer_view(dest, placed.alias, address - reinterpret_cast<u64>(placed.owner->own().title_memory), bytes);
            continue;
        }
        bool have = address && bytes && readable_window(address, bytes) >= bytes && frame_copy(address, bytes, &offset);
        // Experiment (X2W_DRAW_NULL_IDENTITY=1): an empty buffer slot reads as repeated identity
        // matrices instead of zeros. The start-up screen's last pass takes its world matrix from a
        // slot the title leaves empty here; why is not known yet.
        static const bool identity_for_null = x2w_local::env_on("X2W_DRAW_NULL_IDENTITY");
        if (!have && identity_for_null) {
            static float identity[64];
            for (int k = 0; k < 64; ++k) identity[k] = (k % 16) % 5 == 0 ? 1.0f : 0.0f;
            address = reinterpret_cast<u64>(identity);
            bytes = sizeof(identity);
            have = frame_copy(address, bytes, &offset);
        }
        buffer_view(dest, have ? upload_ring : nullptr, offset, bytes);
        count_buffer(have);
        if (first_calls(tr->logged_bindings, 8)) {
            char head[3 * 48 + 1] = "";
            if (have) hex_text(reinterpret_cast<const void*>(address), bytes < 48 ? bytes : 48, head, sizeof(head));
            log("x2w-draw   %016llx srv %u buffer source %d(%d,%d): entry 0x%llx words %08x %08x %08x %08x -> address 0x%llx, %zu bytes, stride %u: %s [%s]",
                tr->key, i, srv.source, srv.a, srv.b, entry, words[0], words[1], words[2], words[3], address, bytes, strides[i],
                have ? "bound" : "NULL", head);
        }
    }
    static LONG checks;
    if ((InterlockedIncrement(&checks) & 63) == 1) check_device("translated draw views", 0);
    std::uint32_t values[32] = {};
    for (std::size_t i = 0; i < tr->uds.size(); ++i) {
        const Translation::Ud& ud = tr->uds[i];
        values[i] = ud.kind == Translation::rect ? (rect ? 1u : 0u)
            : ud.kind == Translation::base_vertex ? static_cast<std::uint32_t>((indexed ? base_vertex : static_cast<INT>(start)) - vertex_first)
            : ud.kind == Translation::start_instance ? start_instance
            : ud.kind == Translation::root_constant ? ((ud.a >= 0 && ud.a < 16 && ud.b >= 0 && ud.b < 8) ? pass.constants[ud.a][ud.b] : 0)
            : ud.kind == Translation::stride && ud.a >= 0 && ud.a < 32 ? strides[ud.a] : 0;
    }
    // A desktop list with no viewport or scissor draws nothing: cover the first render target unless the title set its own.
    if (pass.target0 && (!pass.viewport_set || !pass.scissor_set)) {
        const D3D12_RESOURCE_DESC target = pass.target0->GetDesc();
        // Only the title's own calls set the flags; an automatic default is regenerated when render target 0 changes size.
        if (!pass.viewport_set && (pass.default_viewport_w != target.Width || pass.default_viewport_h != target.Height)) {
            const D3D12_VIEWPORT full{0.0f, 0.0f, static_cast<float>(target.Width), static_cast<float>(target.Height), 0.0f, 1.0f};
            list->RSSetViewports(1, &full);
            pass.default_viewport_w = target.Width;
            pass.default_viewport_h = target.Height;
        }
        if (!pass.scissor_set && (pass.default_scissor_w != target.Width || pass.default_scissor_h != target.Height)) {
            const D3D12_RECT full{0, 0, static_cast<LONG>(target.Width), static_cast<LONG>(target.Height)};
            list->RSSetScissorRects(1, &full);
            pass.default_scissor_w = target.Width;
            pass.default_scissor_h = target.Height;
        }
    }
    list->SetPipelineState(pipeline);
    // X2W_DEPTH_STATE: the reference of the decoded stencil test is list state in D3D12. It replaces the one a
    // preceding OMSetStencilRef set, for draws of this pipeline only when a depth view is bound.
    const bool scoped_stencil_ref = bound->depth_from_registers && bound->has_stencil_ref &&
                                    pass_dsv_format(*bound, pass) != DXGI_FORMAT_UNKNOWN &&
                                    bound->stencil_ref != pass.stencil_reference;
    if (scoped_stencil_ref) list->OMSetStencilRef(bound->stencil_ref);
    list->SetGraphicsRootSignature(tr->root);
    ID3D12DescriptorHeap* samplers = tr->samplers.heap ? pass.sampler_heap.Get() : nullptr;
    if (!pass.heap_set || pass.bound_sampler_heap != samplers) {
        x2w_startup::bind_translated_heaps(list, draw_heap, samplers);
        pass.heap_set = true;
        pass.bound_sampler_heap = samplers;
    }
    list->SetGraphicsRoot32BitConstants(0, static_cast<UINT>(tr->uds.size()), values, 0);
    if (n) list->SetGraphicsRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{draw_heap->GetGPUDescriptorHandleForHeapStart().ptr + u64{first} * draw_heap_increment});
    if (samplers) list->SetGraphicsRootDescriptorTable(x2w_startup::sampler_root_parameter(n), samplers->GetGPUDescriptorHandleForHeapStart());
    list->IASetPrimitiveTopology(pass.topology == 5 ? D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    bool drawn = false;
    if (indexed) {
        u64 offset = 0;
        const u64 address = index_address;
        const std::size_t bytes = index_total;
        if (indices_ok && frame_copy(address, bytes, &offset)) {
            const D3D12_INDEX_BUFFER_VIEW view{upload_ring->GetGPUVirtualAddress() + offset, static_cast<UINT>(bytes),
                                               index_bytes == 2 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT};
            list->IASetIndexBuffer(&view);
            list->DrawIndexedInstanced(count, instances ? instances : 1, 0, 0, start_instance);
            drawn = true;
        }
    } else {
        list->DrawInstanced(rect ? count / 3 * 6 : count, instances ? instances : 1, 0, start_instance);
        drawn = true;
    }
    if (scoped_stencil_ref) list->OMSetStencilRef(pass.stencil_reference);
    if (drawn) draw_dump_after(self, list, tr->key, count);
    ReleaseSRWLockExclusive(&draw_lock);
    if (first_calls(tr->logged_draws, 10) || (count > 100 && first_calls(tr->logged_large, 12))) {
        char indices[3 * 24 + 1] = "";
        if (indexed && pass.ib.BufferLocation) hex_text(reinterpret_cast<const void*>(pass.ib.BufferLocation + u64{start} * 2), 24, indices, sizeof(indices));
        log("x2w-draw %s(%u, %u, start %u, base %d) pipeline %016llx topology 0x%x -> %s; entry [%x %x %x %x %x %x]; ib 0x%llx+%u fmt %d [%s]; "
            "vb 0x%llx %u/%u; targets %u", indexed ? "indexed" : "draw", count, instances, start, base_vertex, tr->key, pass.topology,
            drawn ? "drawn" : "skipped (index data)", values[0], values[1], values[2], values[3], values[4], values[5], pass.ib.BufferLocation,
            pass.ib.SizeInBytes, pass.ib.Format, indices, pass.vb[0].BufferLocation, pass.vb[0].SizeInBytes, pass.vb[0].StrideInBytes,
            pass.target_count);
    }
    return drawn;
}

// List slot 32, SetDescriptorHeaps. Track the title's shader-visible sampler
// source while translated draws use their own physical descriptor heaps.
void __stdcall list_set_descriptor_heaps(Wrapper* self, UINT count, void* const* heaps)
{
    x2w_startup::SamplerSource source;
    u64 view_heap = 0;
    if (count && count <= 4 && x2w_local::readable(heaps, std::size_t{count} * sizeof(*heaps))) {
        AcquireSRWLockShared(&note_lock);
        for (UINT i = 0; i < count; ++i) {
            for (const HeapInfo& heap : heap_infos()) {
                if (heap.wrapper == reinterpret_cast<u64>(heaps[i]) &&
                    heap.type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV &&
                    heap.logical_count > heap.count)
                    view_heap = heap.wrapper;
                if (heap.wrapper == reinterpret_cast<u64>(heaps[i]) &&
                    heap.type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER && heap.desktop_gpu)
                    source = {static_cast<std::uintptr_t>(heap.title_sampler_cpu), heap.count};
            }
        }
        ReleaseSRWLockShared(&note_lock);
    }
    auto& pass = pass_of(self);
    if (pass.source_view_heap != view_heap)
        for (auto& table : pass.view_tables) table = {};
    pass.source_view_heap = view_heap;
    pass.title_samplers = source;
}

// List slot 36 (desktop 32 + 4), SetGraphicsRootDescriptorTable(parameter, GPU handle): the table's
// address in the title's descriptor memory is kept for the draw.
void __stdcall list_table(Wrapper* self, UINT parameter, u64 gpu_handle)
{
    if (parameter >= 16) return;
    auto& pass = pass_of(self);
    x2w_startup::ViewTable table;
    AcquireSRWLockShared(&note_lock);
    for (const HeapInfo& heap : heap_infos()) {
        if (heap.type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) continue;
        if (pass.source_view_heap && heap.wrapper != pass.source_view_heap) continue;
        if (!pass.source_view_heap && heap.logical_count > heap.count) continue;
        table = x2w_startup::resolve_view_table(heap.desktop_gpu, heap.title_cpu, heap.logical_count, heap.increment, gpu_handle);
        if (table.address) break;
    }
    ReleaseSRWLockShared(&note_lock);
    pass.view_tables[parameter] = table;
}
// List slot 40 (desktop 36 + 4), SetGraphicsRoot32BitConstants(parameter, count, data, offset).
void __stdcall list_root_constants(Wrapper* self, UINT parameter, UINT count, const std::uint32_t* data, UINT offset)
{
    static LONG dropped;
    if (count && (parameter >= 16 || offset >= 8 || count > 8 - offset) && first_calls(dropped, 4))
        log("x2w-render root constants %s: parameter %u, offset %u, count %u (16 parameters of 8 words are kept)",
            parameter >= 16 || offset >= 8 ? "dropped" : "truncated", parameter, offset, count);
    if (parameter >= 16 || offset >= 8 || !count || !x2w_local::readable(data, std::size_t{count} * 4)) return;
    for (UINT i = 0; i < count && offset + i < 8; ++i) pass_of(self).constants[parameter][offset + i] = data[i];
}
// List slot 47 (desktop 43 + 4), IASetIndexBuffer(view).
void __stdcall list_index_buffer(Wrapper* self, const D3D12_INDEX_BUFFER_VIEW* view)
{
    pass_of(self).ib = view && x2w_local::readable(view, sizeof(*view)) ? *view : D3D12_INDEX_BUFFER_VIEW{};
}
// List slot 48 (desktop 44 + 4), IASetVertexBuffers(start, count, views).
void __stdcall list_vertex_buffers(Wrapper* self, UINT start, UINT count, const D3D12_VERTEX_BUFFER_VIEW* views)
{
    if (!views || !x2w_local::readable(views, std::size_t{count} * sizeof(*views))) return;
    if (start >= 8) return;
    Pass& pass = pass_of(self);
    for (UINT i = 0; i < count && i < 8 - start; ++i) {
        pass.vb[start + i] = views[i];
        pass.placed_vb[start + i] = placed_buffer_range(self->own().list_device, views[i].BufferLocation, views[i].SizeInBytes);
    }
}
// List slot 13, DrawIndexedInstanced(index count, instances, start index, base vertex, start instance).
void __stdcall list_draw_indexed(Wrapper* self, UINT count, UINT instances, UINT start, INT base_vertex, UINT start_instance)
{
    InterlockedIncrement(draw_translated(self, true, count, instances, start, base_vertex, start_instance) ? &translated_draws : &untranslated_draws);
}
