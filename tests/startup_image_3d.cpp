// Desktop readback for tools/startup/gfx2hlsl.py's 3D image sampling (dim:SQ_RSRC_IMG_3D, #1149).
//
// The translated pixel shader samples r0 : register(t0), a Texture3D<float4>, through q0 :
// register(s0) at (0.75, 0.25, z), with z taken from an entry constant (cbuffer entry : register(b0),
// ud1). This draws one pixel with that shader and a source-built full-screen vertex shader, then
// reads the colour back.
//
// The texture is 2x2x2 R32G32B32A32_FLOAT. Texel (x, y, z) holds ((1 + x + 2y + 4z) / 8, x, y, z),
// so the readback names the texel the coordinate selected. By default it uses the all-zero inline
// sampler (point wrap). The inline sampler fixture passes its own manifest line for the same draw.
#include "desktop_gpu.h"

#include "startup_stubs/translated_sampler_bindings.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT entry_constants = 4;  // ud0..ud3 of the fixture's manifest
constexpr UINT side = 2;
// The mip fixture: a 4x4 array of two slices with three mip levels (4x4, 2x2, 1x1).
constexpr UINT mip_base = 4;
constexpr UINT max_levels = 3;

// The value at texel (x, y) of slice s and mip level m. One level keeps the 2x2x2 fixture's values. With
// several levels every (slice, mip, x, y) cell is distinct: the red channel is a dyadic fraction and
// alpha is s + 2m, so exchanging slice and mip, or reading another level, changes the colour.
void texel_value(UINT x, UINT y, UINT s, UINT m, UINT levels, float out[4])
{
    if (levels == 1) {
        const float one[4] = {(1.0f + x + 2.0f * y + 4.0f * s) / 8.0f, float(x), float(y), float(s)};
        std::memcpy(out, one, sizeof(one));
        return;
    }
    const float many[4] = {(1.0f + x + 4.0f * y + 16.0f * s + 32.0f * m) / 128.0f, float(x), float(y), float(s + 2 * m)};
    std::memcpy(out, many, sizeof(many));
}

std::vector<char> read_blob(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (!file.good() && !file.eof()) throw std::runtime_error(std::string("Cannot read ") + path);
    if (bytes.empty()) throw std::runtime_error(std::string("Empty shader ") + path);
    return bytes;
}

ComPtr<ID3D12RootSignature> create_root(ID3D12Device* device, const x2w_startup::TranslatedSamplers& samplers)
{
    ComPtr<ID3D12RootSignature> root;
    check(x2w_startup::create_translated_root(device, entry_constants, 1, samplers, &root),
          "Create translated image 3D root signature");
    return root;
}
}  // namespace

// array: the same eight texels as two slices of a 2D array (dim:SQ_RSRC_IMG_2D_ARRAY); the default
// view of a two-slice texture is a Texture2DArray over both. levels (array only, 1 or 3): the array is
// 4x4 with that many mip levels, and the default view spans all of them. When sampler_line is a
// comparison sampler (smpc), the texture uses DXGI_FORMAT_R32_FLOAT holding the red channel as depth.
void check_startup_image_3d(ID3D12Device* device, const char* vertex_path, const char* pixel_path, float z,
                            float color[4], const char* sampler_line, bool array, UINT levels, bool one_dimensional,
                            bool two_dimensional)
{
    if (levels < 1 || levels > max_levels || (levels > 1 && !array))
        throw std::runtime_error("Unsupported image mip level count");
    const UINT base = levels > 1 ? mip_base : side;
    const auto vs = read_blob(vertex_path);
    const auto ps = read_blob(pixel_path);
    x2w_startup::TranslatedSamplers samplers;
    if (!samplers.read_manifest_line(sampler_line) || samplers.statics.size() != 1)
        throw std::runtime_error("Expected one inline sampler in the image 3D draw");
    const bool comparison = !samplers.comparison.empty() && samplers.comparison[0];
    const auto root = create_root(device, samplers);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.Get();
    pd.VS = {vs.data(), vs.size()};
    pd.PS = {ps.data(), ps.size()};
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    pd.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pipeline)), "Create image 3D PSO");

    D3D12_RESOURCE_DESC td{};
    td.Dimension = one_dimensional ? D3D12_RESOURCE_DIMENSION_TEXTURE1D
                 : (array || two_dimensional) ? D3D12_RESOURCE_DIMENSION_TEXTURE2D
                 : D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    td.Width = base;
    td.Height = one_dimensional ? 1 : base;
    td.DepthOrArraySize = (one_dimensional || two_dimensional) ? 1 : side;
    td.MipLevels = levels;
    td.Format = comparison ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R32G32B32A32_FLOAT;
    td.SampleDesc.Count = 1;
    const auto texture = resource(device, td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    // A volume is one subresource of two slices; an array is two subresources of one slice each.
    // The subresource of slice s and mip m is m + s * levels.
    const UINT subresources = array ? side * levels : 1;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[side * max_levels]{};
    UINT64 upload_size = 0;
    device->GetCopyableFootprints(&td, 0, subresources, 0, footprints, nullptr, nullptr, &upload_size);
    const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint = footprints[0];
    const auto upload = make_buffer(device, upload_size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    const D3D12_RANGE none{0, 0};
    check(upload->Map(0, &none, &mapped), "Map image 3D texels");
    auto* const start = static_cast<std::uint8_t*>(mapped);
    const UINT slice_pitch = footprint.Footprint.RowPitch * footprint.Footprint.Height;
    for (UINT tz = 0; tz < td.DepthOrArraySize; ++tz)
        for (UINT level = 0; level < levels; ++level) {
            const UINT extent = base >> level;
            const auto& place = array ? footprints[level + tz * levels] : footprint;
            for (UINT ty = 0; ty < (one_dimensional ? 1u : extent); ++ty)
                for (UINT tx = 0; tx < extent; ++tx) {
                    float texel[4];
                    texel_value(tx, ty, tz, level, levels, texel);
                    if (one_dimensional) texel[3] = tx ? 0.75f : 0.5f;
                    auto* const slice = array ? start + place.Offset : start + place.Offset + tz * slice_pitch;
                    const UINT texel_bytes = comparison ? sizeof(float) : sizeof(texel);
                    std::memcpy(slice + ty * place.Footprint.RowPitch + tx * texel_bytes, texel, texel_bytes);
                }
        }
    upload->Unmap(0, nullptr);

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = 1;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R32G32B32A32_FLOAT, {-1.0f, -1.0f, -1.0f, -1.0f}};
    const auto target = resource(device, rd, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT target_footprint{};
    UINT64 readback_size = 0;
    device->GetCopyableFootprints(&rd, 0, 1, 0, &target_footprint, nullptr, nullptr, &readback_size);
    const auto readback = make_buffer(device, readback_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> views, rtvs;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)), "Create image 3D view heap");
    device->CreateShaderResourceView(texture.Get(), nullptr, views->GetCPUDescriptorHandleForHeapStart());
    hd = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtvs)), "Create image 3D RTV heap");
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);

    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    for (UINT i = 0; i < subresources; ++i) {
        D3D12_TEXTURE_COPY_LOCATION to{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        to.SubresourceIndex = i;
        D3D12_TEXTURE_COPY_LOCATION from{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        from.PlacedFootprint = footprints[i];
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
    transition(list, texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
    list->SetGraphicsRootSignature(root.Get());
    ID3D12DescriptorHeap* heaps[]{views.Get()};
    list->SetDescriptorHeaps(1, heaps);
    // ud0 is the rectangle-list switch (0: plain triangles), ud1 the z coordinate's bits.
    std::uint32_t z_bits;
    std::memcpy(&z_bits, &z, sizeof(z_bits));
    const std::uint32_t constants[entry_constants] = {0, z_bits, 0, 0};
    list->SetGraphicsRoot32BitConstants(0, entry_constants, constants, 0);
    list->SetGraphicsRootDescriptorTable(1, views->GetGPUDescriptorHandleForHeapStart());
    const D3D12_VIEWPORT viewport{0, 0, 1, 1, 0, 1};
    const D3D12_RECT scissor{0, 0, 1, 1};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION out{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    out.PlacedFootprint = target_footprint;
    D3D12_TEXTURE_COPY_LOCATION in{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&out, 0, 0, 0, &in, nullptr);
    commands.submit_and_wait(device);

    const D3D12_RANGE read{0, 4 * sizeof(float)};
    check(readback->Map(0, &read, &mapped), "Map image 3D readback");
    std::memcpy(color, static_cast<std::uint8_t*>(mapped) + target_footprint.Offset, 4 * sizeof(float));
    readback->Unmap(0, &none);
    std::printf("startup_image_%s z=%g color=(%g, %g, %g, %g)\n",
                one_dimensional ? "1d" : two_dimensional ? "2d" : array ? "2d_array" : "3d",
                z, color[0], color[1], color[2], color[3]);
}
