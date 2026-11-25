// Resource GetDesc (desktop slot 10) on console resource wrappers.
//
//   startup_resource_desc <warp|hardware>     (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1 X2W_D3D12_DEBUG=1)
//
// Through D3D12XboxCreateDevice the probe creates committed buffer and texture resources
// and queries their descriptions through ID3D12Resource::GetDesc() (slot 10).
//
// Exit codes: 0 pass, 77 no physical adapter, other values name the failing step.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0, Word f = 0, Word g = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e, f, g);
}
bool ok(Word hr) { return SUCCEEDED(static_cast<HRESULT>(hr)); }

std::atomic<unsigned long long> debug_error_count{0};
void WINAPI on_message(D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID, LPCSTR text, void*)
{
    if (severity > D3D12_MESSAGE_SEVERITY_ERROR) return;
    if (++debug_error_count <= 12) std::printf("debug layer: %.300s\n", text);
}

struct Probe {
    ComPtr<ID3D12Device> desktop;
    DWORD cookie = 0;
    ComPtr<ID3D12InfoQueue1> queue1;
    bool watch()
    {
        return SUCCEEDED(desktop.As(&queue1)) &&
               SUCCEEDED(queue1->RegisterMessageCallback(on_message, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie));
    }
};
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    ComPtr<IDXGIAdapter> adapter;
    if (std::strcmp(argv[1], "hardware") == 0) {
        for (UINT n = 0;; ++n) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT hr = factory->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (hr == DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(hr)) return 3;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc))) return 3;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                adapter = candidate;
                break;
            }
        }
    } else if (std::strcmp(argv[1], "warp") != 0 || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        return 3;
    }
    DXGI_ADAPTER_DESC adapter_desc{};
    if (FAILED(adapter->GetDesc(&adapter_desc))) return 3;
    std::printf("adapter=%ls\n", adapter_desc.Description);

    Probe probe;
    std::array<unsigned char, 64> parameters{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &device))) return 4;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&probe.desktop)))) return 4;

    if (!probe.watch()) std::printf("no debug layer message callback: validation not counted\n");

    // 1. Create committed buffer (64 KiB) through the startup device wrapper (slot 27).
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC buffer_desc{
        D3D12_RESOURCE_DIMENSION_BUFFER, 0, 65536, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE
    };
    void* buffer_res = nullptr;
    if (!ok(invoke(device, 27, arg(&heap), D3D12_HEAP_FLAG_NONE, arg(&buffer_desc),
                   D3D12_RESOURCE_STATE_COMMON, 0, arg(&__uuidof(ID3D12Resource)), arg(&buffer_res))))
        return 5;

    // 2. Create committed texture (128x64 RGBA8) through the startup device wrapper (slot 27).
    const D3D12_RESOURCE_DESC tex_desc{
        D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 128, 64, 1, 1,
        DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
    };
    const D3D12_CLEAR_VALUE rtv_clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 0}};
    void* tex_res = nullptr;
    if (!ok(invoke(device, 27, arg(&heap), D3D12_HEAP_FLAG_NONE, arg(&tex_desc),
                   D3D12_RESOURCE_STATE_COMMON, arg(&rtv_clear), arg(&__uuidof(ID3D12Resource)), arg(&tex_res))))
        return 6;

    // 3. Query buffer description via ID3D12Resource::GetDesc() virtual call.
    auto* cl_buf = static_cast<ID3D12Resource*>(buffer_res);
    const D3D12_RESOURCE_DESC got_buf = cl_buf->GetDesc();
    std::printf("1 buffer desc: dim=%d width=%llu height=%u mips=%u fmt=%d layout=0x%x\n",
                got_buf.Dimension, static_cast<unsigned long long>(got_buf.Width),
                got_buf.Height, got_buf.MipLevels, got_buf.Format, got_buf.Layout);
    if (got_buf.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || got_buf.Width != 65536 ||
        got_buf.Height != 1 || got_buf.DepthOrArraySize != 1 || got_buf.MipLevels != 1 ||
        got_buf.Format != DXGI_FORMAT_UNKNOWN || got_buf.Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR)
        return 10;

    // 4. Query texture description via ID3D12Resource::GetDesc() virtual call.
    auto* cl_tex = static_cast<ID3D12Resource*>(tex_res);
    const D3D12_RESOURCE_DESC got_tex = cl_tex->GetDesc();
    std::printf("2 texture desc: dim=%d width=%llu height=%u mips=%u fmt=%d flags=0x%x\n",
                got_tex.Dimension, static_cast<unsigned long long>(got_tex.Width),
                got_tex.Height, got_tex.MipLevels, got_tex.Format, got_tex.Flags);
    if (got_tex.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || got_tex.Width != 128 ||
        got_tex.Height != 64 || got_tex.DepthOrArraySize != 1 || got_tex.MipLevels != 1 ||
        got_tex.Format != DXGI_FORMAT_R8G8B8A8_UNORM || got_tex.Flags != D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
        return 11;

    // 5. Query through function pointer with swapped arguments to verify `receiver` handles both ABI orders.
    using GetDescSlot = void* (WINAPI*)(void*, void*);
    auto get_desc_fn = reinterpret_cast<GetDescSlot>((*static_cast<void***>(tex_res))[10]);
    D3D12_RESOURCE_DESC swapped{};
    void* ret = get_desc_fn(&swapped, tex_res);
    if (ret != &swapped || swapped.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || swapped.Width != 128 ||
        swapped.Height != 64)
        return 12;

    // 6. Negative control: call with null destination pointer returns null without crashing.
    void* null_call = get_desc_fn(tex_res, nullptr);
    if (null_call != nullptr) return 13;

    // 7. Verify device health and debug layer
    const HRESULT removed = probe.desktop->GetDeviceRemovedReason();
    const unsigned long long errors = debug_error_count.load();
    std::printf("device removed reason: 0x%08lx, debug layer errors: %llu\n", static_cast<unsigned long>(removed),
                static_cast<unsigned long long>(errors));
    return removed == S_OK && errors == 0 ? 0 : 70;
}
