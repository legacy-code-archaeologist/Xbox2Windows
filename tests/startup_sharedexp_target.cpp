// Create the shared-exponent target and views through the startup device, then
// clear and read the backing texture on the desktop device.
#include "desktop_gpu.h"
#include "startup_stubs/packed_target_format.h"
#include "startup_stubs/placed_build.h"

#include <d3d12sdklayers.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;

namespace {
constexpr auto sharedexp = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
constexpr auto hdr = DXGI_FORMAT_R11G11B10_FLOAT;
constexpr UINT width = 4, height = 2;
constexpr float rt_color[]{0.5f, 1.0f, 2.0f, 1.0f};
constexpr float uav_color[]{-1.0f, 0.5f, 996.0f, 1.0f};

// The fixture's positive inputs are finite normal values away from rounding ties.
// The biased exponent is 15 for values in [1, 2); quantize the significand at
// that exponent, then pack the red, green and blue fields into a little-endian word.
std::uint32_t encode_channel(float value, int mantissa_bits)
{
    if (value <= 0.0f) return 0;
    int exponent;
    const double significand = std::frexp(value, &exponent);
    const auto rounded = static_cast<unsigned>(std::round(std::ldexp(significand, mantissa_bits + 1)));
    return ((exponent + 14u) << mantissa_bits) + rounded - (1u << mantissa_bits);
}

std::uint32_t encode_rgb(const float (&color)[4])
{
    return encode_channel(color[0], 6) | (encode_channel(color[1], 6) << 11) |
           (encode_channel(color[2], 5) << 22);
}

ComPtr<ID3D12Resource> native_texture(ID3D12Device* device, ID3D12Device* guest, ID3D12Resource* texture)
{
    // The device wrapper unwraps objects passed to SetPrivateDataInterface.
    constexpr GUID key{0x029d497b, 0xc94c, 0x42ca, {0xa1, 0xad, 0x0e, 0xe3, 0xa5, 0xd8, 0x66, 0xd1}};
    check(guest->SetPrivateDataInterface(key, texture), "Publish underlying texture");
    ComPtr<IUnknown> stored;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT read = device->GetPrivateData(key, &bytes, stored.GetAddressOf());
    check(guest->SetPrivateDataInterface(key, nullptr), "Clear underlying texture");
    check(read, "Retrieve underlying texture");
    ComPtr<ID3D12Resource> result;
    check(stored.As(&result), "Query underlying texture");
    return result;
}

ComPtr<ID3D12DescriptorHeap> heap(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type, UINT count,
                                 D3D12_DESCRIPTOR_HEAP_FLAGS flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE)
{
    const D3D12_DESCRIPTOR_HEAP_DESC desc{type, count, flags};
    ComPtr<ID3D12DescriptorHeap> result;
    check(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&result)), "Create fixture descriptor heap");
    return result;
}

template <class Create>
void create_view(ID3D12InfoQueue* messages, Create&& create, const char* stage)
{
    const UINT64 before = messages ? messages->GetNumStoredMessagesAllowedByRetrievalFilter() : 0;
    create();
    if (messages && messages->GetNumStoredMessagesAllowedByRetrievalFilter() != before)
        throw std::runtime_error(stage);  // offscreen_check prints the debug diagnostics
}

void check_texture(ID3D12Device* device, ID3D12Device* guest, ID3D12InfoQueue* messages, UINT16 slices)
{
    const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
    const auto flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, slices, 1, sharedexp,
                                  {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, flags};
    const D3D12_CLEAR_VALUE clear{sharedexp, {rt_color[0], rt_color[1], rt_color[2], rt_color[3]}};
    ComPtr<ID3D12Resource> wrapped;
    check(guest->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                         &clear, IID_PPV_ARGS(&wrapped)), "Create shared-exponent RT|UAV texture");
    const auto texture = native_texture(device, guest, wrapped.Get());
    const auto actual = texture->GetDesc();
    const auto tag = x2w_startup::packed_stand_in_of(wrapped.Get());
    if (actual.Format != hdr || actual.Flags != flags || actual.DepthOrArraySize != slices || tag != sharedexp)
        throw std::runtime_error("Shared-exponent resource shape, actual format or tag differs");
    const auto targets = heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1);
    const auto views = heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2);
    const auto shader_views = heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                  D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
    const auto rtv_handle = targets->GetCPUDescriptorHandleForHeapStart();
    const auto uav_handle = views->GetCPUDescriptorHandleForHeapStart();
    const auto shader_cpu = shader_views->GetCPUDescriptorHandleForHeapStart();
    const auto shader_gpu = shader_views->GetGPUDescriptorHandleForHeapStart();
    const D3D12_CPU_DESCRIPTOR_HANDLE srv_handle{
        uav_handle.ptr + device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)};

    const bool array = slices > 1;
    D3D12_RENDER_TARGET_VIEW_DESC rtv{sharedexp, array ? D3D12_RTV_DIMENSION_TEXTURE2DARRAY : D3D12_RTV_DIMENSION_TEXTURE2D};
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{sharedexp, array ? D3D12_UAV_DIMENSION_TEXTURE2DARRAY : D3D12_UAV_DIMENSION_TEXTURE2D};
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{sharedexp, array ? D3D12_SRV_DIMENSION_TEXTURE2DARRAY : D3D12_SRV_DIMENSION_TEXTURE2D,
                                        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
    srv.Texture2D.MipLevels = 1;
    if (array) {
        rtv.Texture2DArray.ArraySize = slices;
        uav.Texture2DArray.ArraySize = slices;
        srv.Texture2DArray.ArraySize = slices;
    }
    create_view(messages, [&] { guest->CreateRenderTargetView(wrapped.Get(), &rtv, rtv_handle); },
                "Create format-67 RTV reported a debug diagnostic");
    create_view(messages, [&] {
        guest->CreateUnorderedAccessView(wrapped.Get(), nullptr, &uav, uav_handle);
        guest->CreateUnorderedAccessView(wrapped.Get(), nullptr, &uav, shader_cpu);
    }, "Create format-67 UAV reported a debug diagnostic");
    create_view(messages, [&] { guest->CreateShaderResourceView(wrapped.Get(), &srv, srv_handle); },
                "Create format-67 SRV reported a debug diagnostic");

    std::array<D3D12_PLACED_SUBRESOURCE_FOOTPRINT, 2> footprints{};
    UINT64 bytes;
    device->GetCopyableFootprints(&actual, 0, slices, 0, footprints.data(), nullptr, nullptr, &bytes);
    auto readback = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ComputeCommands commands(device, nullptr, D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    list->ClearRenderTargetView(rtv_handle, rt_color, 0, nullptr);
    transition(list, texture.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12DescriptorHeap* const bound[] = {shader_views.Get()};
    list->SetDescriptorHeaps(1, bound);
    const D3D12_RECT right{width / 2, 0, width, height};
    list->ClearUnorderedAccessViewFloat(shader_gpu, uav_handle, texture.Get(), uav_color, 1, &right);
    transition(list, texture.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    for (UINT slice = 0; slice < slices; ++slice) {
        D3D12_TEXTURE_COPY_LOCATION dst{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint = footprints[slice];
        D3D12_TEXTURE_COPY_LOCATION src{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        src.SubresourceIndex = slice;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    commands.submit_and_wait(device);

    void* data = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(bytes)}, no_writes{};
    check(readback->Map(0, &range, &data), "Map shared-exponent target readback");
    const auto rt_word = encode_rgb(rt_color), uav_word = encode_rgb(uav_color);
    for (UINT slice = 0; slice < slices; ++slice) {
        const auto& footprint = footprints[slice];
        for (UINT y = 0; y < height; ++y) {
            const auto* row = static_cast<const char*>(data) + footprint.Offset + y * footprint.Footprint.RowPitch;
            for (UINT x = 0; x < width; ++x) {
                std::uint32_t word;
                std::memcpy(&word, row + x * sizeof(word), sizeof(word));
                const auto expected = x < width / 2 ? rt_word : uav_word;
                if (word != expected) {
                    readback->Unmap(0, &no_writes);
                    std::fprintf(stderr, "slice %u (%u,%u): expected %08x, got %08x\n", slice, x, y, expected, word);
                    throw std::runtime_error("Shared-exponent stand-in readback differs");
                }
            }
        }
    }
    readback->Unmap(0, &no_writes);
    std::printf("slices=%u format=%d tag=%d rtv=%08x uav=%08x words=%u\n",
                slices, actual.Format, tag, rt_word, uav_word, width * height * slices);
}
}  // namespace

const char* check_startup_sharedexp_target(ID3D12Device* device, ID3D12InfoQueue* messages)
{
    constexpr UINT uses = D3D12_FORMAT_SUPPORT1_RENDER_TARGET | D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW;
    D3D12_FEATURE_DATA_FORMAT_SUPPORT original{sharedexp}, fallback{hdr};
    check(check_feature(device, D3D12_FEATURE_FORMAT_SUPPORT, &original, sizeof(original)), "Query format 67");
    check(check_feature(device, D3D12_FEATURE_FORMAT_SUPPORT, &fallback, sizeof(fallback)), "Query format 26");
    std::printf("support1_67=0x%x support1_26=0x%x support2_26=0x%x\n", original.Support1, fallback.Support1, fallback.Support2);
    if ((original.Support1 & uses) == uses) return "adapter supports native format-67 RT and UAV views";
    if ((fallback.Support1 & uses) != uses || !(fallback.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE))
        return "adapter lacks format-26 render-target or typed UAV stores";

    device->AddRef();
    void* wrapped = nullptr;
    x2w_startup::attach_device(device, &wrapped);
    ComPtr<ID3D12Device> guest;
    guest.Attach(static_cast<ID3D12Device*>(wrapped));
    check_texture(device, guest.Get(), messages, 1);
    check_texture(device, guest.Get(), messages, 2);
    return nullptr;
}
