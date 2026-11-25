#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <wrl/client.h>

#include <array>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "startup_stream_draw.h"
#include "startup_stream_setup.h"

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

// A slot-55 patch is a tagged union with room for the full eight-target blend state.
struct Patch { UINT tag; D3D12_BLEND_DESC blend; };
static_assert(sizeof(Patch) == 0x14c);

static bool derive(void* device, void*& pipeline, bool child)
{
    using Create = HRESULT (WINAPI*)(void*, void*, UINT, const Patch*, REFIID, void**);
    auto create = reinterpret_cast<Create>((*static_cast<void***>(device))[55]);
    Patch preserve{2}, write{2}, sample_mask{3};
    const UINT all_samples = ~0u;
    std::memcpy(&sample_mask.blend, &all_samples, sizeof(all_samples));
    auto& t = preserve.blend.RenderTarget[0];
    t = {TRUE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_BLEND_ZERO,
         D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
    write.blend = preserve.blend;
    write.blend.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    write.blend.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    void* next = nullptr;
    if (create(device, pipeline, 1, nullptr, __uuidof(ID3D12PipelineState), &next) != E_INVALIDARG || next) return false;
    if (FAILED(create(device, pipeline, 1, &preserve, __uuidof(ID3D12PipelineState), &next)) || !next) return false;
    pipeline = next;
    if (child) {
        Patch patches[] = {preserve, write, sample_mask};
        if (FAILED(create(device, pipeline, 3, patches, __uuidof(ID3D12PipelineState), &next)) || !next) return false;
        pipeline = next;
    }
    // A grandchild with only a non-blend patch keeps its nearest ancestor's state.
    if (FAILED(create(device, pipeline, 1, &sample_mask, __uuidof(ID3D12PipelineState), &next)) || !next) return false;
    pipeline = next;
    return true;
}

// A slot-57 description as the stand-in reads it: 64-bit words 0 (root signature), 1-2 (state block
// A and its size), 5-6 and 9-10 (the two shaders). spec lists "offset:word" pairs in hex, written
// into a zeroed state block A; the colour-blend words of the eight targets start at +0xD0.
// The shaders and the root signature are read from the files the stream pipeline was made from, so
// both pipelines share one translation.
static bool create_slot57(void* device, const char* vs_path, const char* ps_path, const char* root_path, const char* spec,
                          void*& pipeline)
{
    static std::vector<char> vs, ps;
    std::ifstream vs_file(vs_path, std::ios::binary), ps_file(ps_path, std::ios::binary), root_file(root_path, std::ios::binary);
    vs.assign(std::istreambuf_iterator<char>(vs_file), {});
    ps.assign(std::istreambuf_iterator<char>(ps_file), {});
    const std::vector<char> blob{std::istreambuf_iterator<char>(root_file), {}};
    if (vs.empty() || ps.empty() || blob.empty()) return false;
    using CreateRoot = HRESULT (WINAPI*)(void*, UINT, const void*, SIZE_T, REFIID, void**);
    void* root = nullptr;
    if (FAILED(reinterpret_cast<CreateRoot>((*static_cast<void***>(device))[16])(device, 0, blob.data(), blob.size(),
                                                                                  __uuidof(ID3D12RootSignature), &root))) return false;
    alignas(8) static unsigned char desc[0x300];
    alignas(8) static unsigned char state_a[0x200];
    for (const char* at = spec; *at;) {
        char* end = nullptr;
        const unsigned long offset = std::strtoul(at, &end, 16);
        if (*end != ':' || offset % 4 || offset + 4 > sizeof(state_a)) return false;
        const UINT word = static_cast<UINT>(std::strtoul(end + 1, &end, 16));
        std::memcpy(state_a + offset, &word, sizeof(word));
        if (*end && *end != ',') return false;
        at = *end ? end + 1 : end;
    }
    const UINT64 words[12] = {reinterpret_cast<UINT64>(root), reinterpret_cast<UINT64>(state_a), sizeof(state_a), 0, 0,
                              reinterpret_cast<UINT64>(vs.data()), vs.size(), 0, 0, reinterpret_cast<UINT64>(ps.data()), ps.size(), 0};
    std::memcpy(desc, words, sizeof(words));
    using Create = HRESULT (WINAPI*)(void*, const void*, REFIID, void**);
    pipeline = nullptr;
    return SUCCEEDED(reinterpret_cast<Create>((*static_cast<void***>(device))[57])(device, desc, __uuidof(ID3D12PipelineState), &pipeline)) &&
           pipeline;
}

// Binds the pipeline on a guest list and records `calls` ExecuteIndirect calls (list slot 61) with a Draw or a
// Dispatch signature. No root or target is bound, so every call is refused and nothing is drawn.
static bool bind_and_execute_indirect(void* device, void* pipeline, bool dispatch, int calls)
{
    auto* guest = static_cast<ID3D12Device*>(device);
    D3D12_INDIRECT_ARGUMENT_DESC argument{};
    argument.Type = dispatch ? D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH : D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
    const D3D12_COMMAND_SIGNATURE_DESC signature_desc{
        dispatch ? UINT{sizeof(D3D12_DISPATCH_ARGUMENTS)} : UINT{sizeof(D3D12_DRAW_ARGUMENTS)}, 1, &argument, 0};
    ComPtr<ID3D12CommandSignature> signature;
    ComPtr<ID3D12CommandAllocator> allocator;
    if (FAILED(guest->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature))) ||
        FAILED(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)))) return false;
    const auto slots = [](void* object) { return *static_cast<void***>(object); };
    using CreateList = HRESULT (WINAPI*)(void*, const void*, void*, void*, REFIID, void**);
    const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    void* list = nullptr;
    if (FAILED(reinterpret_cast<CreateList>(slots(device)[73])(device, list_desc, allocator.Get(), nullptr,
                                                                 __uuidof(ID3D12GraphicsCommandList), &list)) || !list) return false;
    using Reset = HRESULT (WINAPI*)(void*, void*, void*, void*);
    using Bind = void (WINAPI*)(void*, void*);
    using Execute = void (WINAPI*)(void*, void*, UINT, void*, UINT64, void*, UINT64);
    using Close = HRESULT (WINAPI*)(void*);
    if (FAILED(reinterpret_cast<Reset>(slots(list)[139])(list, allocator.Get(), nullptr, nullptr))) return false;
    reinterpret_cast<Bind>(slots(list)[25])(list, pipeline);
    for (int i = 0; i < calls; ++i) reinterpret_cast<Execute>(slots(list)[61])(list, signature.Get(), 1, nullptr, 0, nullptr, 0);
    const bool closed = SUCCEEDED(reinterpret_cast<Close>(slots(list)[9])(list));
    static_cast<IUnknown*>(list)->Release();
    return closed;
}

// A slot-86 pipeline with desktop shaders that use no resources, over the stream root: a desktop pipeline.
static void* desktop_pipeline(void* device)
{
    static const char vs_text[] = "float4 main(uint id : SV_VertexID) : SV_Position { return float4(0, 0, 0, 1); }";
    static const char ps_text[] = "float4 main() : SV_Target { return float4(1, 0, 0, 1); }";
    ComPtr<ID3DBlob> vs, ps;
    if (FAILED(D3DCompile(vs_text, sizeof(vs_text) - 1, nullptr, nullptr, nullptr, "main", "vs_5_0", 0, 0, &vs, nullptr)) ||
        FAILED(D3DCompile(ps_text, sizeof(ps_text) - 1, nullptr, nullptr, nullptr, "main", "ps_5_0", 0, 0, &ps, nullptr)))
        return nullptr;
    struct alignas(void*) Root { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; void* value; };
    struct alignas(void*) Code { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; D3D12_SHADER_BYTECODE value; };
    struct alignas(void*) Formats { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; D3D12_RT_FORMAT_ARRAY value; };
    struct alignas(void*) Topology { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; D3D12_PRIMITIVE_TOPOLOGY_TYPE value; };
    struct { Root root; Code vs, ps; Formats targets; Topology topology; } stream{
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, stream_root_signature()},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, {vs->GetBufferPointer(), vs->GetBufferSize()}},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, {ps->GetBufferPointer(), ps->GetBufferSize()}},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, {{DXGI_FORMAT_R8G8B8A8_UNORM}, 1}},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE}};
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
    using Create = HRESULT (WINAPI*)(void*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
    void* pipeline = nullptr;
    return SUCCEEDED(reinterpret_cast<Create>((*static_cast<void***>(device))[86])(device, &desc, __uuidof(ID3D12PipelineState),
                                                                                   &pipeline)) ? pipeline : nullptr;
}

int main(int argc, char** argv)
{
    if (argc != 4 && (argc < 6 || argc > 8) && argc != 10) return 2;
    const bool draw = argc != 4;
    StartupBc7Input bc7{};
    if (argc == 10) {
        bc7 = {argv[6], static_cast<UINT>(std::strtoul(argv[7], nullptr, 10)),
               static_cast<UINT>(std::strtoul(argv[8], nullptr, 10)),
               std::strcmp(argv[9], "tiled") == 0};
        if ((bc7.width != 64 || bc7.height != 64) && (bc7.width != 81 || bc7.height != 69)) return 2;
        if (std::strncmp(argv[9], "packed:", 7) == 0) bc7.packed = argv[9] + 7;
        else if (!bc7.tiled && std::strcmp(argv[9], "linear") != 0) return 2;
    }
    ComPtr<IDXGIAdapter> adapter;
    void* device = nullptr;
    void* pso = nullptr;
    const bool hardware = draw && std::strcmp(argv[4], "hardware") == 0;
    if (const int rc = prepare_stream_pipeline(argv[1], argv[2], argv[3], hardware, bc7.blocks != nullptr,
                                                adapter, device, pso)) return rc;
    if (GetEnvironmentVariableW(L"X2W_TEST_TRUNCATED_ONLY", nullptr, 0)) return 0;
    if (argc == 4) return 0;
    DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
    UINT export_word = 0;
    bool two_targets = false;
    if (argc == 7 && std::strncmp(argv[6], "indirect-", 9) == 0) {
        // A pipeline used only through slot 61. slot57-draw: three refused Draw calls; slot57-bound: bound, no call;
        // slot57-dispatch: one refused Dispatch call; desktop-draw: a desktop pipeline and one refused Draw call.
        const std::string mode(argv[6] + 9);
        if (mode == "desktop-draw") pso = desktop_pipeline(device);
        else if (mode == "slot57-draw" || mode == "slot57-bound" || mode == "slot57-dispatch") {
            if (!create_slot57(device, argv[1], argv[2], argv[3], "", pso)) return 10;
        } else return 2;
        if (!pso) return 10;
        const int calls = mode == "slot57-draw" ? 3 : mode == "slot57-bound" ? 0 : 1;
        return bind_and_execute_indirect(device, pso, mode == "slot57-dispatch", calls) ? 0 : 11;
    }
    if (argc == 8 && std::strncmp(argv[6], "slot57", 6) == 0) {
        // The same shaders and root through device slot 57, with the given words in state block A.
        // "slot57-mrt" binds a second RGBA8 target.
        two_targets = std::strcmp(argv[6], "slot57-mrt") == 0;
        if (!two_targets && std::strcmp(argv[6], "slot57") != 0) return 2;
        if (!create_slot57(device, argv[1], argv[2], argv[3], argv[7], pso)) return 10;
    } else if (argc == 7 && !std::isdigit(static_cast<unsigned char>(argv[6][0]))) {
        const std::string mode(argv[6]);
        if (mode != "parent-a8" && mode != "child-a8" && mode != "child-r8") return 2;
        if (!derive(device, pso, mode != "parent-a8")) return 10;
        format = mode == "child-r8" ? DXGI_FORMAT_R8_UNORM : DXGI_FORMAT_A8_UNORM;
    } else if (argc == 7 || argc == 8) {
        // A render-target format for the translated MRT export, and an optional root constant.
        format = static_cast<DXGI_FORMAT>(std::strtoul(argv[6], nullptr, 10));
        if (format != DXGI_FORMAT_R32_UINT && format != DXGI_FORMAT_R32_SINT &&
            format != DXGI_FORMAT_R32G32B32A32_FLOAT) return 2;
        if (argc == 8) export_word = static_cast<UINT>(std::strtoul(argv[7], nullptr, 0));
    }
    wchar_t manifest[MAX_PATH];
    const DWORD length = GetEnvironmentVariableW(L"X2W_TEST_PUBLISH_MANIFEST", manifest, MAX_PATH);
    if (length && length < MAX_PATH) {
        const std::string first = std::string(argv[5]) + ".first";
        if (const int rc = check_stream_draw(adapter.Get(), device, pso, first.c_str(), nullptr, format)) return rc;
        if (!MoveFileW((std::wstring(manifest) + L".pending").c_str(), manifest)) return 61;
        Sleep(1100);
    }
    if (GetEnvironmentVariableW(L"X2W_TEST_PLACED_VERTEX", nullptr, 0)) {
        extern void release_placed_vertices(bool);
        const std::string first = std::string(argv[5]) + ".first";
        if (const int rc = check_stream_draw(adapter.Get(), device, pso, first.c_str(), nullptr, format)) return rc;
        // GPU fence and readback are complete before the high range is released.
        release_placed_vertices(false);
        const int rc = check_stream_draw(adapter.Get(), device, pso, argv[5], nullptr, format);
        release_placed_vertices(true);
        return rc;
    }
    return check_stream_draw(adapter.Get(), device, pso, argv[5], bc7.blocks ? &bc7 : nullptr, format, export_word, two_targets);
}
