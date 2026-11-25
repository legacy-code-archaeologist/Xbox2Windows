#pragma once

#include "xmem_section_lease.h"

#include <d3d12.h>
#include <wrl/client.h>

namespace x2w_startup {

// The caller supplies the exact shader-accessible range and holds this object
// through the last GPU consumer. A section window is backing, never a proof
// that its other bytes may be accessed by that shader.
class XMemSectionBuffer {
public:
    XMemSectionBuffer() = default;
    XMemSectionBuffer(const XMemSectionBuffer&) = delete;
    XMemSectionBuffer& operator=(const XMemSectionBuffer&) = delete;
    ~XMemSectionBuffer()
    {
        buffer.Reset();
        heap_.Reset();
        if (lease.section) CloseHandle(lease.section);
        // A view that could not be removed still has access to the slots.
        if ((!view_ || UnmapViewOfFile(view_)) && lease.id) release_(version_, lease.id);
    }

    bool matches(std::uint64_t candidate) const
    {
        return match_ && match_(version_, lease.id, lease.first_byte, candidate, bytes_);
    }

    D3D12_GPU_VIRTUAL_ADDRESS gpu_address() const
    {
        const auto base = buffer ? buffer->GetGPUVirtualAddress() : 0;
        return base && lease.first_byte <= UINT64_MAX - base ? base + lease.first_byte : 0;
    }

    // Failure is final for the instance. The destructor releases any backing
    // acquired on the way. The overlap predicate sees the physical lease
    // before D3D12 maps it as a resource.
    template<class Accept>
    bool open(ID3D12Device* device, UINT version, std::uint64_t address, std::uint64_t bytes, Accept accept)
    {
        if (lease.id || !device || !address || !bytes || bytes > SIZE_MAX || bytes > UINT64_MAX - address)
            return false;
        HMODULE xmem{};
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, L"xmem.dll", &xmem)) return false;
        using Pin = BOOL (WINAPI*)(UINT, ULONG_PTR, SIZE_T, X2WXMemSectionLease*, SIZE_T);
        const auto pin = reinterpret_cast<Pin>(GetProcAddress(xmem, "x2w_xmem_lease_section"));
        release_ = reinterpret_cast<Release>(GetProcAddress(xmem, "x2w_xmem_release_section"));
        match_ = reinterpret_cast<X2WXMemMatchSection>(GetProcAddress(xmem, "x2w_xmem_match_section"));
        if (!pin || !release_ || !match_) return false;

        Microsoft::WRL::ComPtr<ID3D12Device3> desktop;
        if (FAILED(device->QueryInterface(IID_PPV_ARGS(&desktop)))) return false;
        D3D12_FEATURE_DATA_EXISTING_HEAPS feature{};
        const auto check_heaps = [&] {
#if defined(X2W_MESH_XMEM_AGILITY_620) && defined(__ID3D12Device16_INTERFACE_DEFINED__)
            Microsoft::WRL::ComPtr<ID3D12Device16> versioned;
            if (SUCCEEDED(desktop.As(&versioned)))
                return versioned->CheckFeatureSupport1(D3D12_FEATURE_EXISTING_HEAPS, &feature, sizeof(feature), 620);
#endif
            return desktop->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &feature, sizeof(feature));
        };
        if (FAILED(check_heaps()) || !feature.Supported) return false;
        version_ = version;
        bytes_ = bytes;
        if (!pin(version_, address, static_cast<SIZE_T>(bytes), &lease, sizeof(lease)) ||
            !lease.section || !lease.id || !lease.window_bytes || lease.window_bytes > SIZE_MAX ||
            lease.first_byte > lease.window_bytes || bytes > lease.window_bytes - lease.first_byte ||
            lease.window_bytes > UINT64_MAX - lease.section_offset || !accept(lease)) return false;
        view_ = MapViewOfFile(lease.section, FILE_MAP_WRITE, static_cast<DWORD>(lease.section_offset >> 32),
                              static_cast<DWORD>(lease.section_offset), static_cast<SIZE_T>(lease.window_bytes));
        if (!view_ || FAILED(desktop->OpenExistingHeapFromAddress(view_, IID_PPV_ARGS(&heap_)))) return false;
        const D3D12_RESOURCE_DESC desc{
            D3D12_RESOURCE_DIMENSION_BUFFER, 0, lease.window_bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
            {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER
        };
        return SUCCEEDED(desktop->CreatePlacedResource(heap_.Get(), 0, &desc, D3D12_RESOURCE_STATE_COMMON,
            nullptr, IID_PPV_ARGS(&buffer))) && gpu_address() && matches(address);
    }

    X2WXMemSectionLease lease{};
    Microsoft::WRL::ComPtr<ID3D12Resource> buffer;

private:
    using Release = BOOL (WINAPI*)(UINT, ULONG64);
    Release release_{};
    X2WXMemMatchSection match_{};
    UINT version_{};
    std::uint64_t bytes_{};
    void* view_{};
    Microsoft::WRL::ComPtr<ID3D12Heap> heap_;
};

} // namespace x2w_startup
