// Experimental (docs/startup-runtime.md): stand-in formats for packed render and unordered-access targets that
// the adapter cannot use the way the description asks.
//
// A console executable creates B5G5R5A1_UNORM and B4G4R4A4_UNORM 2D textures as render targets,
// the first also with unordered access. B5G6R5_UNORM and 2D arrays (a cube map is an array of six
// slices) follow the same rule. An adapter may not support that (measured: neither format
// as a render target or typed UAV on an NVIDIA adapter through vkd3d-proton); the description is
// then refused, the allocation-info answer is all ones and the executable gets no texture. Such a
// texture is created as R8G8B8A8_UNORM and tagged with the format it stands in for; a view that
// names that format on that texture is created as R8G8B8A8_UNORM.
//
// R9G9B9E5_SHAREDEXP can use R11G11B10_FLOAT for a render target, a typed UAV, or both. Both
// use 32 bits per texel and clamp negative values to zero. R11G11B10_FLOAT has 6/6/5 mantissa
// bits and per-channel exponents; format 67 has three 9-bit mantissas with one shared exponent.
// Uploads and copies in the original shared-exponent bit layout are not converted.
//
// For 16-bit formats the stand-in keeps 8 bits a channel where the packed format keeps 5, 6, 4 or 1; a
// B5G6R5_UNORM stand-in also has an alpha channel the packed format lacks (it samples as 1): values the
// executable renders are not quantized, so a read of the target can differ from the console's in
// the low bits (0.45 reads 115 here, 119 from B4G4R4A4). Texel data the executable writes or reads
// itself in the 16-bit layout (uploads, copies to or from buffers) is not converted; only textures
// that are rendered to and then sampled or presented are covered.
#pragma once

#include <d3d12.h>

namespace x2w_startup {
struct PackedTargetMapping {
    DXGI_FORMAT requested;
    DXGI_FORMAT replacement;
    UINT trigger_flags;
};

inline constexpr PackedTargetMapping packed_target_mappings[] = {
    {DXGI_FORMAT_B5G5R5A1_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET},
    {DXGI_FORMAT_B4G4R4A4_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET},
    {DXGI_FORMAT_B5G6R5_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET},
    {DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R11G11B10_FLOAT,
        UINT{D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET} | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS},
};

inline const PackedTargetMapping* packed_target_mapping(DXGI_FORMAT format)
{
    for (const auto& mapping : packed_target_mappings)
        if (mapping.requested == format) return &mapping;
    return nullptr;
}

inline bool packed_target_format(DXGI_FORMAT format)
{
    return packed_target_mapping(format) != nullptr;
}

// The format to create a texture with. support1(format) returns the adapter's
// D3D12_FORMAT_SUPPORT1 bits for a format, or 0 when the query fails. For a single-sample 2D
// texture or array the mapping's trigger flags enable the check. Substitute only if the adapter
// lacks a requested target use for the original format and the replacement supports them all.
//
// An array of 2 to D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION slices is covered the same way: the
// stand-in changes the format and nothing else, so slice count, mip count and view dimensions of
// the description and of its views are kept. Zero slices, more than the limit, three dimensions and
// multisampling are not modelled and keep their format.
template <class Support1>
DXGI_FORMAT packed_target_stand_in(const D3D12_RESOURCE_DESC& desc, Support1&& support1)
{
    const auto* mapping = packed_target_mapping(desc.Format);
    if (!mapping || !(desc.Flags & mapping->trigger_flags) ||
        desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.DepthOrArraySize == 0 ||
        desc.DepthOrArraySize > D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION || desc.SampleDesc.Count != 1 ||
        (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))
        return desc.Format;
    UINT required = 0;
    if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) required |= D3D12_FORMAT_SUPPORT1_RENDER_TARGET;
    if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) required |= D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW;
    if ((static_cast<UINT>(support1(desc.Format)) & required) == required) return desc.Format;
    if ((static_cast<UINT>(support1(mapping->replacement)) & required) != required) return desc.Format;
    return mapping->replacement;
}

// Provenance: a texture created as a stand-in carries the format it was asked for as private data.
// {6B1D3C52-8E0A-4F1B-9C6E-2A7D5E4F9B31}
inline constexpr GUID packed_stand_in_tag = {0x6b1d3c52, 0x8e0a, 0x4f1b, {0x9c, 0x6e, 0x2a, 0x7d, 0x5e, 0x4f, 0x9b, 0x31}};

// Tags `created` when it is a stand-in for `requested` (the description's desktop format).
inline void mark_packed_stand_in(ID3D12Resource* created, DXGI_FORMAT requested)
{
    const auto* mapping = packed_target_mapping(requested);
    if (created && mapping && created->GetDesc().Format == mapping->replacement)
        created->SetPrivateData(packed_stand_in_tag, sizeof(requested), &requested);
}

// The packed format `resource` stands in for, or DXGI_FORMAT_UNKNOWN.
inline DXGI_FORMAT packed_stand_in_of(ID3D12Resource* resource)
{
    DXGI_FORMAT requested = DXGI_FORMAT_UNKNOWN;
    UINT bytes = sizeof(requested);
    if (!resource || FAILED(resource->GetPrivateData(packed_stand_in_tag, &bytes, &requested)) || bytes != sizeof(requested))
        return DXGI_FORMAT_UNKNOWN;
    return requested;
}

enum class PackedView {
    unchanged,  // not a packed view, or a packed view of a texture of that very format
    stand_in,   // use the stand-in texture's format
    refused,    // a packed view of a texture with no compatible format
};

// What to do with a view described with view_format on a texture of resource_format that stands in
// for stands_in_for (DXGI_FORMAT_UNKNOWN: an ordinary texture).
inline PackedView packed_view(DXGI_FORMAT view_format, DXGI_FORMAT resource_format, DXGI_FORMAT stands_in_for)
{
    const auto* mapping = packed_target_mapping(view_format);
    if (!mapping) return PackedView::unchanged;
    if (stands_in_for == DXGI_FORMAT_UNKNOWN) return resource_format == view_format ? PackedView::unchanged : PackedView::refused;
    return stands_in_for == view_format && resource_format == mapping->replacement ? PackedView::stand_in : PackedView::refused;
}
}  // namespace x2w_startup
