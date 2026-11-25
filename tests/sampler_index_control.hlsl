// The two index controls differ only in the source of the sampler descriptor index.
// The split-sampler readback also uses this shader to check for missing output stores.
//
// The descriptor tables, the shader model, the output indexing and the three sampler swaps stay
// identical between the two builds, so a difference in the readback can only come from the index:
//   INDEX_FROM_GROUP_ID undefined : index = SV_DispatchThreadID.x  (the passing control)
//   INDEX_FROM_GROUP_ID defined   : index = SV_GroupID.x           (the recorded reproduction)
//
// The bindings match tests/spirv_window/sampler_readback.cpp: t0 is a 2x1 R32_FLOAT texture holding
// 0.25 and 0.75, u0 is a four-word raw buffer, s1-s2 in space 0 are point samplers with wrap and
// clamp U addressing, and s1-s2 in space 3 are LESS_EQUAL comparison samplers. Word i holds the
// ordinary sample taken with descriptor i, and word i + 2 the comparison taken with descriptor i.
Texture2D<float4> depth_texture : register(t0);
RWByteAddressBuffer results : register(u0);
SamplerState ordinary[2] : register(s1);
SamplerComparisonState comparison[2] : register(s1, space3);

[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread : SV_DispatchThreadID, uint3 group : SV_GroupID)
{
#if defined(INDEX_FROM_GROUP_ID)
    const uint sampler_index = group.x;
#else
    const uint sampler_index = dispatch_thread.x;
#endif
    const uint output_index = dispatch_thread.x;
    const float2 uv = float2(1.25, 0.5);
    const float sample_value = depth_texture.SampleLevel(ordinary[sampler_index], uv, 0).x;
    results.Store(output_index * 4, asuint(sample_value));
#if defined(OMIT_STORES_ON_ORDINARY_SWAP)
    // Skip both comparison stores only when the ordinary sampler pair is reversed.
    if (sample_value != (output_index == 0 ? 0.25 : 0.75)) return;
#endif
    results.Store((output_index + 2) * 4,
                  asuint(depth_texture.SampleCmpLevelZero(comparison[sampler_index], uv, 0.5)));
}
