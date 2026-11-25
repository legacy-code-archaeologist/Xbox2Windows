#include "../desktop_gpu.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

using Microsoft::WRL::ComPtr;

void check_sampler_views(ID3D12Device* device, const char* path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open sampler shader");
    std::vector<char> shader(std::istreambuf_iterator<char>(stream), {});
    D3D12_DESCRIPTOR_RANGE ranges[] = {
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 2, 1, 0, 0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 2, 1, 3, 0},
    };
    D3D12_ROOT_PARAMETER params[4]{};
    for (UINT i = 0; i < 4; ++i) {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[i].DescriptorTable = {1, &ranges[i]};
    }
    D3D12_ROOT_SIGNATURE_DESC desc{4, params};
    ComPtr<ID3DBlob> bytes, error;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &bytes, &error), "Serialize sampler root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, bytes->GetBufferPointer(), bytes->GetBufferSize(), IID_PPV_ARGS(&root)), "Sampler root");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.Get();
    pd.CS = {shader.data(), shader.size()};
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pipeline)), "Sampler pipeline");

    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = 2;
    td.Height = td.DepthOrArraySize = td.MipLevels = td.SampleDesc.Count = 1;
    td.Format = DXGI_FORMAT_R32_FLOAT;
    auto texture = resource(device, td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 upload_size{};
    device->GetCopyableFootprints(&td, 0, 1, 0, &footprint, nullptr, nullptr, &upload_size);
    auto upload = make_buffer(device, upload_size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    const float depths[] = {0.25f, 0.75f};
    void* mapped = nullptr;
    D3D12_RANGE empty{};
    check(upload->Map(0, &empty, &mapped), "Map depth values");
    std::memcpy(mapped, depths, sizeof(depths));
    upload->Unmap(0, nullptr);

    auto output = make_buffer(device, 16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback = make_buffer(device, 16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    const std::array<float, 4> sentinels{-1.0f, -2.0f, -3.0f, -4.0f};
    auto reset = make_buffer(device, sizeof(sentinels), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    check(reset->Map(0, &empty, &mapped), "Map sampler fixture sentinels");
    std::memcpy(mapped, sentinels.data(), sizeof(sentinels));
    reset->Unmap(0, nullptr);
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> views, samplers;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)), "Create sampler fixture views");
    auto cpu = views->GetCPUDescriptorHandleForHeapStart();
    device->CreateShaderResourceView(texture.Get(), nullptr, cpu);
    UINT view_stride = device->GetDescriptorHandleIncrementSize(hd.Type);
    cpu.ptr += view_stride;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Format = DXGI_FORMAT_R32_TYPELESS;
    uav.Buffer.NumElements = 4;
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    device->CreateUnorderedAccessView(output.Get(), nullptr, &uav, cpu);
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    hd.NumDescriptors = 4;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&samplers)), "Create split sampler tables");
    UINT sampler_stride = device->GetDescriptorHandleIncrementSize(hd.Type);

    // Reverse each typed pair independently. The other operation must retain its results.
    bool mismatched = false;
    for (UINT mask : {0u, 1u, 2u}) {
        auto handle = samplers->GetCPUDescriptorHandleForHeapStart();
        for (UINT i = 0; i < 4; ++i) {
            bool comparison = i >= 2;
            bool clamp = ((i & 1) != 0) ^ ((mask & (comparison ? 2 : 1)) != 0);
            D3D12_SAMPLER_DESC sampler{};
            sampler.Filter = comparison ? D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT : D3D12_FILTER_MIN_MAG_MIP_POINT;
            sampler.AddressU = clamp ? D3D12_TEXTURE_ADDRESS_MODE_CLAMP : D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            sampler.ComparisonFunc = comparison ? D3D12_COMPARISON_FUNC_LESS_EQUAL : D3D12_COMPARISON_FUNC_NEVER;
            sampler.MaxAnisotropy = 1;
            device->CreateSampler(&sampler, handle);
            handle.ptr += sampler_stride;
        }
        ComputeCommands commands(device, pipeline.Get());
        auto list = commands.list.Get();
        if (mask == 0) {
            D3D12_TEXTURE_COPY_LOCATION dst{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            D3D12_TEXTURE_COPY_LOCATION src{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            src.PlacedFootprint = footprint;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            transition(list, texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        // Each sampler swap must write its own four results, including values unchanged from the previous swap.
        transition(list, output.Get(), mask == 0 ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(output.Get(), 0, reset.Get(), 0, sizeof(sentinels));
        transition(list, output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetComputeRootSignature(root.Get());
        ID3D12DescriptorHeap* heaps[] = {views.Get(), samplers.Get()};
        list->SetDescriptorHeaps(2, heaps);
        auto gpu = views->GetGPUDescriptorHandleForHeapStart();
        list->SetComputeRootDescriptorTable(0, gpu);
        gpu.ptr += view_stride;
        list->SetComputeRootDescriptorTable(1, gpu);
        gpu = samplers->GetGPUDescriptorHandleForHeapStart();
        list->SetComputeRootDescriptorTable(2, gpu);
        gpu.ptr += 2 * sampler_stride;
        list->SetComputeRootDescriptorTable(3, gpu);
        list->Dispatch(2, 1, 1);
        transition(list, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, 16);
        commands.submit_and_wait(device);
        std::array<float, 4> expected{0.25f, 0.75f, 0.0f, 1.0f};
        if (mask == 1) std::swap(expected[0], expected[1]);
        if (mask == 2) std::swap(expected[2], expected[3]);
        D3D12_RANGE read{0, 16};
        check(readback->Map(0, &read, &mapped), "Map sampler fixture results");
        std::array<float, 4> actual;
        std::memcpy(actual.data(), mapped, sizeof(actual));
        readback->Unmap(0, &empty);
        std::printf("sampler swap %u: sample %g %g, comparison %g %g\n",
                    mask, actual[0], actual[1], actual[2], actual[3]);
        for (size_t i = 0; i < expected.size(); ++i) {
            UINT expected_word, actual_word;
            std::memcpy(&expected_word, &expected[i], sizeof(expected_word));
            std::memcpy(&actual_word, &actual[i], sizeof(actual_word));
            if (actual_word != expected_word) {
                std::printf("sampler swap %u word %zu: expected=%08x actual=%08x\n",
                            mask, i, expected_word, actual_word);
                mismatched = true;
            }
        }
    }
    if (mismatched) throw std::runtime_error("Split sampler readback mismatch");
}
