#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "desktop_gpu.h"
#include "startup_indirect_mesh_mesh.h"
#include "startup_indirect_mesh_pixel.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Microsoft::WRL::ComPtr;
using Words = std::array<std::uint32_t, 7>;
constexpr UINT width = 4, height = 2;
constexpr auto format = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr std::uint32_t black = 0xff000000, red = 0xff0000ff, green = 0xff00ff00;
constexpr Words initial{0xa5a5a5a5, 0, 0, 0, 0, 0, 0x5a5a5a5a};
constexpr std::array<std::uint16_t, 10> indices{99, 99, 99, 0, 1, 2, 3, 4, 5, 99};
constexpr std::array<std::uint32_t, 10> wide_indices{99, 99, 99, 0, 1, 2, 3, 4, 5, 99};
constexpr UINT argument_offset = sizeof(std::uint32_t);

constexpr char shader[] = R"(
RWByteAddressBuffer data : register(u0);

[numthreads(1, 1, 1)]
void write_left() {
    data.Store4(4, uint4(3, 1, 1, 0));
    data.Store(20, 0);
}

[numthreads(1, 1, 1)]
void write_right() {
    data.Store4(4, uint4(3, 1, 4, 0));
    data.Store(20, 0);
}

[numthreads(1, 1, 1)]
void write_one_triangle() {
    data.Store4(4, uint4(3, 1, 0, 0));
}

[numthreads(1, 1, 1)]
void write_two_triangles() {
    data.Store4(4, uint4(6, 1, 0, 0));
}

[numthreads(1, 1, 1)]
void write_one_mesh_group() {
    data.Store3(4, uint3(1, 1, 1));
}

[numthreads(1, 1, 1)]
void write_two_mesh_groups() {
    data.Store3(4, uint3(2, 1, 1));
}

struct Varying {
    float4 position : SV_Position;
    nointerpolation uint side : SIDE;
};

Varying vertex(uint id : SV_VertexID) {
    Varying result;
    uint corner = id % 3;
    result.position = float4(corner == 2 ? 3 : -1, corner == 1 ? 3 : -1, 0, 1);
    result.side = id / 3;
    return result;
}

float4 pixel(Varying input) : SV_Target {
    if (input.side == 0 && input.position.x < 2) return float4(1, 0, 0, 1);
    if (input.side == 1 && input.position.x >= 2) return float4(0, 1, 0, 1);
    discard;
    return float4(0, 0, 0, 1);
}
)";

template <typename Method>
Method slot(IUnknown* object, unsigned index)
{
    return reinterpret_cast<Method>((*reinterpret_cast<void***>(object))[index]);
}

template <typename T>
ComPtr<T> native_object(ID3D12Device* device, ID3D12Device* guest, IUnknown* object)
{
    // The guest forwards SetPrivateDataInterface with the underlying desktop argument.
    constexpr GUID key{0x18097577, 0x0a4a, 0x427c, {0xac, 0xd5, 0xac, 0x1d, 0x98, 0x90, 0x65, 0x85}};
    check(guest->SetPrivateDataInterface(key, object), "publish underlying object");
    ComPtr<IUnknown> stored;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT read = device->GetPrivateData(key, &bytes, stored.GetAddressOf());
    check(guest->SetPrivateDataInterface(key, nullptr), "clear underlying object");
    check(read, "retrieve underlying object");
    ComPtr<T> result;
    check(stored.As(&result), "query underlying object");
    return result;
}

ComPtr<ID3DBlob> compile(const char* entry, const char* profile)
{
    ComPtr<ID3DBlob> code, errors;
    const HRESULT hr = D3DCompile(shader, sizeof(shader) - 1, nullptr, nullptr, nullptr, entry, profile, 0, 0,
                                  &code, &errors);
    if (FAILED(hr) && errors) std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
    check(hr, entry);
    return code;
}

ComPtr<ID3D12RootSignature> create_root(ID3D12Device* device, const D3D12_ROOT_SIGNATURE_DESC& desc)
{
    ComPtr<ID3DBlob> code;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &code, nullptr), "serialize root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, code->GetBufferPointer(), code->GetBufferSize(), IID_PPV_ARGS(&root)),
          "create root");
    return root;
}

ComPtr<ID3D12PipelineState> writer_pipeline(ID3D12Device* device, ID3D12RootSignature* root, const char* entry)
{
    const auto cs = compile(entry, "cs_5_0");
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
    ComPtr<ID3D12PipelineState> result;
    check(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&result)), "create writer pipeline");
    return result;
}

template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Kind, typename T>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = Kind;
    T value{};
};

template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Stage>
ComPtr<IUnknown> guest_pipeline(ID3D12Device* guest, ID3D12RootSignature* root,
                               D3D12_SHADER_BYTECODE source, D3D12_SHADER_BYTECODE pixel)
{
    struct {
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
        Subobject<Stage, D3D12_SHADER_BYTECODE> source;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC> depth;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> targets;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, DXGI_SAMPLE_DESC> samples;
    } stream;
    static_assert(sizeof(stream) == 16 + 24 + 24 + 48 + 56 + 8 + 40 + 16);
    stream.root.value = root;
    stream.source.value = source;
    stream.ps.value = pixel;
    stream.raster.value.FillMode = D3D12_FILL_MODE_SOLID;
    stream.raster.value.CullMode = D3D12_CULL_MODE_NONE;
    stream.raster.value.DepthClipEnable = TRUE;
    stream.depth.value.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    stream.depth.value.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    stream.topology.value = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    stream.targets.value.RTFormats[0] = format;
    stream.targets.value.NumRenderTargets = 1;
    stream.samples.value.Count = 1;
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
    using Create = HRESULT (WINAPI*)(IUnknown*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
    ComPtr<IUnknown> result;
    check(slot<Create>(guest, 86)(guest, &desc, __uuidof(ID3D12PipelineState),
                                 reinterpret_cast<void**>(result.GetAddressOf())), "create guest graphics pipeline");
    return result;
}

struct FreeMemory {
    void operator()(void* pointer) const { VirtualFree(pointer, 0, MEM_RELEASE); }
};

struct PlacedBuffer {
    PlacedBuffer(ID3D12Device* device, ID3D12Device* guest, UINT width, D3D12_RESOURCE_FLAGS flags)
        : backing(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))
    {
        if (!backing) throw std::bad_alloc();
        // A deliberately unaligned placed address selects the runtime's committed stand-in.
        // Its GPU data is independent of the zero-filled guest virtual address.
        address = static_cast<char*>(backing.get()) + 4;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = width;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = flags;
        using Place = HRESULT (WINAPI*)(IUnknown*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, REFIID, void**);
        check(slot<Place>(guest, 133)(guest, address, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, __uuidof(ID3D12Resource),
                                      reinterpret_cast<void**>(wrapped.GetAddressOf())), "place guest buffer");
        native = native_object<ID3D12Resource>(device, guest, wrapped.Get());
    }

    std::unique_ptr<void, FreeMemory> backing;
    void* address{};
    ComPtr<IUnknown> wrapped;
    ComPtr<ID3D12Resource> native;
};

ComPtr<IUnknown> begin_guest_list(ID3D12Device* guest, ID3D12CommandAllocator* allocator)
{
    using CreateList = HRESULT (WINAPI*)(IUnknown*, const void*, IUnknown*, IUnknown*, REFIID, void**);
    const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    ComPtr<IUnknown> result;
    check(slot<CreateList>(guest, 73)(guest, list_desc, allocator, nullptr, __uuidof(ID3D12GraphicsCommandList),
                                     reinterpret_cast<void**>(result.GetAddressOf())), "create guest command list");
    using Reset = HRESULT (WINAPI*)(IUnknown*, IUnknown*, void*, void*);
    check(slot<Reset>(result.Get(), 139)(result.Get(), allocator, nullptr, nullptr), "reset guest command list");
    return result;
}

void submit(ID3D12Device* device, ID3D12GraphicsCommandList* list)
{
    const D3D12_COMMAND_QUEUE_DESC desc{};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)), "create queue");
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "create completion fence");
    ID3D12CommandList* submitted[]{list};
    queue->ExecuteCommandLists(1, submitted);
    check(queue->Signal(fence.Get(), 1), "signal completion fence");
    check(fence->SetEventOnCompletion(1, nullptr), "wait for completion");
    check(device->GetDeviceRemovedReason(), "GPU completion");
}

enum class Refusal { none, invalid_index, short_argument, missing_viewport };
enum class DrawKind { indexed, vertices, mesh };

constexpr UINT command_stride(DrawKind kind)
{
    switch (kind) {
    case DrawKind::indexed: return sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
    case DrawKind::vertices: return sizeof(D3D12_DRAW_ARGUMENTS);
    case DrawKind::mesh: return sizeof(D3D12_DISPATCH_MESH_ARGUMENTS);
    }
    return 0;
}

bool supports_mesh(ID3D12Device* device)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options, sizeof(options))) &&
        options.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1;
}

bool run(ID3D12Device* device, ID3D12Device* guest, ID3D12RootSignature* writer_root, ID3D12PipelineState* writer,
         ID3D12RootSignature* graphics_root, IUnknown* graphics, ID3D12CommandSignature* signature,
         DrawKind kind, bool draw_right, Refusal refusal, const char* label)
{
    const bool indexed = kind == DrawKind::indexed;
    const bool mesh = kind == DrawKind::mesh;
    const UINT index_bytes = indexed ? (draw_right ? sizeof(wide_indices) : sizeof(indices)) : 0;
    const UINT index_offset = 2 * (draw_right ? sizeof(std::uint32_t) : sizeof(std::uint16_t));
    const void* index_data = draw_right ? static_cast<const void*>(wide_indices.data()) : indices.data();
    Words initial_words = initial;
    if (!indexed) initial_words[1 + command_stride(kind) / sizeof(std::uint32_t)] = 0x3c3c3c3c;
    const PlacedBuffer arguments(device, guest, sizeof(initial_words), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    std::optional<PlacedBuffer> index_buffer;
    if (indexed) index_buffer.emplace(device, guest, index_bytes, D3D12_RESOURCE_FLAG_NONE);
    const auto upload = make_buffer(device, sizeof(initial_words) + index_bytes, D3D12_HEAP_TYPE_UPLOAD,
                                    D3D12_RESOURCE_STATE_GENERIC_READ);
    const auto arg_readback = make_buffer(device, sizeof(initial), D3D12_HEAP_TYPE_READBACK,
                                          D3D12_RESOURCE_STATE_COPY_DEST);
    void* memory = nullptr;
    const D3D12_RANGE untouched{0, 0};
    check(upload->Map(0, &untouched, &memory), "map upload");
    std::memcpy(memory, initial_words.data(), sizeof(initial_words));
    if (indexed) std::memcpy(static_cast<char*>(memory) + sizeof(initial_words), index_data, index_bytes);
    upload->Unmap(0, nullptr);

    const D3D12_RESOURCE_DESC target_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, format, {1, 0},
                                          D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    constexpr float clear[]{0, 0, 0, 1};
    const D3D12_CLEAR_VALUE clear_value{format, {0, 0, 0, 1}};
    const auto target = resource(device, target_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, &clear_value);
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1,
                                              D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "create RTV heap");
    const auto rtv = heap->GetCPUDescriptorHandleForHeapStart();
    guest->CreateRenderTargetView(target.Get(), nullptr, rtv);

    ComPtr<ID3D12CommandAllocator> allocator;
    check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "create guest allocator");
    const auto wrapped_list = begin_guest_list(guest, allocator.Get());
    auto* wrapped = wrapped_list.Get();
    const auto native_list = native_object<ID3D12GraphicsCommandList>(device, guest, wrapped);
    auto* list = native_list.Get();
    if (index_buffer) {
        transition(list, index_buffer->native.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(index_buffer->native.Get(), 0, upload.Get(), sizeof(initial_words), index_bytes);
        transition(list, index_buffer->native.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    }
    transition(list, arguments.native.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(arguments.native.Get(), 0, upload.Get(), 0, sizeof(initial));
    auto argument_state = D3D12_RESOURCE_STATE_COPY_DEST;
    if (writer) {
        transition(list, arguments.native.Get(), argument_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        argument_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        list->SetPipelineState(writer);
        list->SetComputeRootSignature(writer_root);
        list->SetComputeRootUnorderedAccessView(0, arguments.native->GetGPUVirtualAddress());
        list->Dispatch(1, 1, 1);
    }
    transition(list, arguments.native.Get(), argument_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(arg_readback.Get(), 0, arguments.native.Get(), 0, sizeof(initial));
    // The wrapper owns the subsequent COMMON -> INDIRECT_ARGUMENT transition.
    transition(list, arguments.native.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);

    using Bind = void (WINAPI*)(IUnknown*, IUnknown*);
    using Clear = void (WINAPI*)(IUnknown*, SIZE_T, const float*, UINT, const D3D12_RECT*);
    using Targets = void (WINAPI*)(IUnknown*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, BOOL,
                                  const D3D12_CPU_DESCRIPTOR_HANDLE*);
    using Topology = void (WINAPI*)(IUnknown*, D3D12_PRIMITIVE_TOPOLOGY);
    using Viewport = void (WINAPI*)(IUnknown*, UINT, const D3D12_VIEWPORT*);
    using Scissor = void (WINAPI*)(IUnknown*, UINT, const D3D12_RECT*);
    using Index = void (WINAPI*)(IUnknown*, const D3D12_INDEX_BUFFER_VIEW*);
    using Execute = void (WINAPI*)(IUnknown*, IUnknown*, UINT, IUnknown*, UINT64, IUnknown*, UINT64);
    const D3D12_VIEWPORT viewport{0, 0, width, height, 0, 1};
    const D3D12_RECT scissor{0, 0, width, height};
    slot<Clear>(wrapped, 52)(wrapped, rtv.ptr, clear, 0, nullptr);
    slot<Targets>(wrapped, 50)(wrapped, 1, &rtv, FALSE, nullptr);
    slot<Bind>(wrapped, 25)(wrapped, graphics);
    slot<Bind>(wrapped, 34)(wrapped, graphics_root);
    if (!mesh) slot<Topology>(wrapped, 20)(wrapped, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    if (refusal != Refusal::missing_viewport) slot<Viewport>(wrapped, 21)(wrapped, 1, &viewport);
    slot<Scissor>(wrapped, 22)(wrapped, 1, &scissor);
    if (index_buffer) {
        const auto address = reinterpret_cast<UINT64>(index_buffer->address);
        const auto start = refusal == Refusal::invalid_index ? index_bytes : index_offset;
        const D3D12_INDEX_BUFFER_VIEW view{address + start, index_bytes - index_offset,
                                         draw_right ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT};
        slot<Index>(wrapped, 47)(wrapped, &view);
    }
    const UINT64 offset = refusal == Refusal::short_argument
        ? sizeof(initial_words) - command_stride(kind) + sizeof(std::uint32_t) : argument_offset;
    slot<Execute>(wrapped, 61)(wrapped, signature, 1, arguments.wrapped.Get(), offset, nullptr, 0);
    check(slot<HRESULT (WINAPI*)(IUnknown*)>(wrapped, 9)(wrapped), "close guest command list");
    submit(device, list);

    const D3D12_RANGE arg_range{0, sizeof(initial)};
    check(arg_readback->Map(0, &arg_range, &memory), "map argument readback");
    Words observed{};
    std::memcpy(observed.data(), memory, sizeof(observed));
    arg_readback->Unmap(0, &untouched);
    Words expected = initial_words;
    if (writer) {
        expected[1] = !indexed && draw_right ? 6 : 3;
        expected[2] = 1;
        if (indexed) expected[3] = draw_right ? 4 : 1;
        if (mesh) {
            expected[1] = draw_right ? 2 : 1;
            expected[3] = 1;
        }
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (observed[i] == expected[i]) continue;
        std::fprintf(stderr, "%s: argument word %zu: got 0x%08x, expected 0x%08x\n", label, i, observed[i], expected[i]);
        return false;
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    device->GetCopyableFootprints(&target_desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    const auto pixel_readback = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ComputeCommands copy(device, nullptr, D3D12_COMMAND_LIST_TYPE_DIRECT);
    transition(copy.list.Get(), target.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION source{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_TEXTURE_COPY_LOCATION destination{pixel_readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    destination.PlacedFootprint = footprint;
    copy.list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    copy.submit_and_wait(device);

    const D3D12_RANGE pixel_range{0, static_cast<SIZE_T>(bytes)};
    check(pixel_readback->Map(0, &pixel_range, &memory), "map pixel readback");
    UINT painted = 0;
    bool pixels_match = true;
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width; ++x) {
            std::uint32_t actual = 0;
            std::memcpy(&actual, static_cast<char*>(memory) + y * footprint.Footprint.RowPitch + x * sizeof(actual),
                        sizeof(actual));
            const bool right_half = x >= width / 2;
            const bool covered = indexed ? right_half == draw_right : !right_half || draw_right;
            const bool drawn = writer && refusal == Refusal::none && covered;
            const auto wanted = drawn ? (right_half ? green : red) : black;
            if (actual != black) ++painted;
            if (actual == wanted) continue;
            std::fprintf(stderr, "%s: pixel (%u,%u): got 0x%08x, expected 0x%08x\n", label, x, y, actual, wanted);
            pixels_match = false;
        }
    pixel_readback->Unmap(0, &untouched);
    std::printf("%s: %u/%u non-clear pixels, argument guards intact\n", label, painted, width * height);
    return pixels_match;
}
// Calls refused before any argument is read: one per counted reason, and one with two commands, which is
// counted as other under its signature's kind. The list is closed and dropped.
void refuse_each_reason(ID3D12Device* guest, ID3D12CommandSignature* draw_indexed)
{
    D3D12_INDIRECT_ARGUMENT_DESC dispatch{};
    dispatch.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    const D3D12_COMMAND_SIGNATURE_DESC dispatch_desc{sizeof(D3D12_DISPATCH_ARGUMENTS), 1, &dispatch, 0};
    ComPtr<ID3D12CommandSignature> dispatch_signature;
    check(guest->CreateCommandSignature(&dispatch_desc, nullptr, IID_PPV_ARGS(&dispatch_signature)), "create guest dispatch signature");
    ComPtr<ID3D12CommandAllocator> allocator;
    check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "create guest allocator");
    const auto list = begin_guest_list(guest, allocator.Get());
    using Execute = void (WINAPI*)(IUnknown*, IUnknown*, UINT, IUnknown*, UINT64, IUnknown*, UINT64);
    const auto execute = slot<Execute>(list.Get(), 61);
    execute(list.Get(), nullptr, 1, nullptr, 0, nullptr, 0);                    // placeholder signature
    execute(list.Get(), draw_indexed, 1, nullptr, 0, nullptr, 0);               // no desktop graphics pipeline bound
    execute(list.Get(), dispatch_signature.Get(), 1, nullptr, 0, nullptr, 0);   // no desktop compute pipeline bound
    execute(list.Get(), draw_indexed, 2, nullptr, 0, nullptr, 0);               // two commands
    check(slot<HRESULT (WINAPI*)(IUnknown*)>(list.Get(), 9)(list.Get()), "close refusal list");
}

// 300 empty console presents (queue slot 32) reach the render-totals heartbeat; nothing is shown.
std::string indirect_line(ID3D12Device* guest, const std::wstring& log_path)
{
    const D3D12_COMMAND_QUEUE_DESC desc{};
    ComPtr<IUnknown> queue;
    check(guest->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(queue.GetAddressOf())),
          "create guest queue");
    using Present = HRESULT (WINAPI*)(IUnknown*, UINT, const void*, const void*);
    const std::array<unsigned char, 0x50> planes{};
    const std::array<unsigned char, 0x40> present{};
    for (int frame = 0; frame < 300; ++frame) slot<Present>(queue.Get(), 32)(queue.Get(), 0, planes.data(), present.data());
    std::ifstream file(log_path, std::ios::binary);
    std::string line, found;
    while (std::getline(file, line))
        if (line.rfind("x2w-render indirect:", 0) == 0) found = line;
    while (!found.empty() && (found.back() == '\r' || found.back() == '\n')) found.pop_back();
    return found;
}
}  // namespace

int main(int argc, char** argv) try
{
    DrawKind kind = DrawKind::indexed;
    // "count-control" is the negative control: it expects one forwarded call too many.
    const bool count_control = argc == 3 && std::strcmp(argv[2], "count-control") == 0;
    if (argc == 3 && std::strcmp(argv[2], "draw") == 0) kind = DrawKind::vertices;
    else if (argc == 3 && std::strcmp(argv[2], "mesh") == 0) kind = DrawKind::mesh;
    else if (argc != 2 && !count_control) return 2;
    const bool indexed = kind == DrawKind::indexed;
    const bool mesh = kind == DrawKind::mesh;
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    if (!warp && std::strcmp(argv[1], "hardware") != 0) return 2;
    ComPtr<ID3D12Debug> debug;
    const HRESULT debug_hr = D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
    if (debug_hr != DXGI_ERROR_SDK_COMPONENT_MISSING) check(debug_hr, "get debug layer");
    if (debug) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory6> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "create DXGI factory");
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "find WARP");
    } else {
        for (UINT index = 0;; ++index) {
            const HRESULT hr = factory->EnumAdapters1(index, adapter.ReleaseAndGetAddressOf());
            if (hr == DXGI_ERROR_NOT_FOUND) {
                std::puts("indirect graphics capability skip: no compatible physical adapter");
                return 77;
            }
            check(hr, "enumerate physical adapters");
            DXGI_ADAPTER_DESC1 desc{};
            check(adapter->GetDesc1(&desc), "get physical adapter description");
            const bool basic_render_driver = desc.VendorId == 0x1414 && desc.DeviceId == 0x008c;
            if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) || basic_render_driver) continue;
            ComPtr<ID3D12Device> probe;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&probe))) &&
                (!mesh || supports_mesh(probe.Get()))) break;
        }
    }
    wchar_t directory[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, directory)) return 4;
    const std::wstring log_path = std::wstring(directory) + L"x2w_indirect_counts_" + std::to_wstring(GetCurrentProcessId()) + L".log";
    if (indexed) {
        DeleteFileW(log_path.c_str());
        SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path.c_str());
        SetEnvironmentVariableA("X2W_PRESENT_SIGNATURE_EVERY", "0");
    }
    DXGI_ADAPTER_DESC1 desc{};
    check(adapter->GetDesc1(&desc), "get selected adapter description");
    std::printf("adapter: %ls, debug layer %s\n", desc.Description, debug ? "enabled" : "unavailable");
    LARGE_INTEGER driver{};
    if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver)))
        std::printf("driver: %u.%u.%u.%u\n", HIWORD(driver.HighPart), LOWORD(driver.HighPart),
                    HIWORD(driver.LowPart), LOWORD(driver.LowPart));

    ComPtr<ID3D12Device> device, guest;
    check(D3D12XboxCreateDevice(adapter.Get(), nullptr, IID_PPV_ARGS(&device)), "create desktop device through d3d12_xs");
    if (mesh && !supports_mesh(device.Get())) {
        std::puts("indirect mesh capability skip: mesh shaders unavailable");
        return 77;
    }
    check(SetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", "1") ? S_OK : HRESULT_FROM_WIN32(GetLastError()), "enable guest");
    check(SetEnvironmentVariableA("X2W_D3D12_LIST_NOOP", "1") ? S_OK : HRESULT_FROM_WIN32(GetLastError()), "enable guest list");
    check(SetEnvironmentVariableA("X2W_D3D12_RENDER", "1") ? S_OK : HRESULT_FROM_WIN32(GetLastError()), "enable recording");
    const std::array<unsigned char, 64> parameters{};
    check(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), IID_PPV_ARGS(&guest)), "create guest device");
    ComPtr<ID3D12InfoQueue> info;
    if (debug) check(device.As(&info), "get debug message queue");
    const D3D12_ROOT_PARAMETER uav{D3D12_ROOT_PARAMETER_TYPE_UAV, {}, D3D12_SHADER_VISIBILITY_ALL};
    const D3D12_ROOT_SIGNATURE_DESC writer_desc{1, &uav, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto root_flags = mesh ? D3D12_ROOT_SIGNATURE_FLAG_NONE : D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    const D3D12_ROOT_SIGNATURE_DESC graphics_desc{0, nullptr, 0, nullptr, root_flags};
    const auto writer = create_root(device.Get(), writer_desc);
    const auto graphics = create_root(guest.Get(), graphics_desc);
    const char* left_entry = indexed ? "write_left" : "write_one_triangle";
    const char* right_entry = indexed ? "write_right" : "write_two_triangles";
    D3D12_INDIRECT_ARGUMENT_DESC argument{};
    argument.Type = indexed ? D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED : D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
    ComPtr<IUnknown> pso;
    if (mesh) {
        left_entry = "write_one_mesh_group";
        right_entry = "write_two_mesh_groups";
        argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
        pso = guest_pipeline<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS>(guest.Get(), graphics.Get(),
            {startup_indirect_mesh_mesh, sizeof(startup_indirect_mesh_mesh)},
            {startup_indirect_mesh_pixel, sizeof(startup_indirect_mesh_pixel)});
    } else {
        const auto vs = compile("vertex", "vs_5_0");
        const auto ps = compile("pixel", "ps_5_0");
        pso = guest_pipeline<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS>(guest.Get(), graphics.Get(),
            {vs->GetBufferPointer(), vs->GetBufferSize()}, {ps->GetBufferPointer(), ps->GetBufferSize()});
    }
    const auto left = writer_pipeline(device.Get(), writer.Get(), left_entry);
    const auto right = writer_pipeline(device.Get(), writer.Get(), right_entry);
    const D3D12_COMMAND_SIGNATURE_DESC signature_desc{command_stride(kind), 1, &argument, 0};
    ComPtr<ID3D12CommandSignature> signature;
    check(guest->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature)), "create guest draw signature");

    const auto verify = [&](ID3D12PipelineState* producer, bool draw_right, Refusal refusal, const char* label) {
        return run(device.Get(), guest.Get(), writer.Get(), producer, graphics.Get(), pso.Get(), signature.Get(),
                   kind, draw_right, refusal, label);
    };
    const char* left_label = indexed ? "left writer" : "one triangle";
    const char* right_label = indexed ? "right writer" : "two triangles";
    if (mesh) {
        left_label = "one mesh group";
        right_label = "two mesh groups";
    }
    if (!verify(left.Get(), false, Refusal::none, left_label) ||
        !verify(right.Get(), true, Refusal::none, right_label) ||
        !verify(nullptr, false, Refusal::none, "omitted writer") ||
        !verify(left.Get(), false, Refusal::short_argument, "short argument range")) return 1;
    if (indexed ? !verify(left.Get(), false, Refusal::invalid_index, "invalid index range")
                : !verify(left.Get(), false, Refusal::missing_viewport, "missing viewport")) return 1;
    if (indexed) {
        // Forwarded: the three unrefused runs. Other reasons: the short argument range, the invalid index range and two commands.
        refuse_each_reason(guest.Get(), signature.Get());
        const std::string line = indirect_line(guest.Get(), log_path);
        DeleteFileW(log_path.c_str());
        const std::string expected = std::string("x2w-render indirect: forwarded: drawindexed ") + (count_control ? "4" : "3") +
            "; refused graphics-pipeline: drawindexed 1; compute-pipeline: dispatch 1; placeholder-signature: other 1; "
            "other: drawindexed 3";
        if (line != expected) {
            std::fprintf(stderr, "indirect line mismatch:\n  got      '%s'\n  expected '%s'\n", line.c_str(), expected.c_str());
            return 1;
        }
        std::printf("%s\n", line.c_str());
    }
    if (info) {
        for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
            SIZE_T bytes = 0;
            check(info->GetMessage(i, nullptr, &bytes), "get debug message size");
            std::vector<char> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            check(info->GetMessage(i, message, &bytes), "get debug message");
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                std::fprintf(stderr, "D3D12 debug message %u: %s\n", static_cast<unsigned>(message->ID), message->pDescription);
                return 1;
            }
        }
    }
    return 0;
}
catch (const std::exception&) {
    return 3;
}
