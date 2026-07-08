#include "ibl_common.hlsli"

TextureCube<float4> gEnvironmentTexture : register(t0, space0);
SamplerState gEnvironmentSampler : register(s2, space0);
RWTexture2DArray<float4> gOutputPrefilter : register(u1, space0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint outputWidth = 0;
    uint outputHeight = 0;
    uint outputLayers = 0;
    gOutputPrefilter.GetDimensions(outputWidth, outputHeight, outputLayers);

    if (dispatchThreadId.x >= outputWidth ||
        dispatchThreadId.y >= outputHeight ||
        dispatchThreadId.z >= min(outputLayers, 6u))
    {
        return;
    }

    uint sourceWidth = 0;
    uint sourceHeight = 0;
    uint sourceLevels = 0;
    gEnvironmentTexture.GetDimensions(0, sourceWidth, sourceHeight, sourceLevels);

    const float maxMip = max(log2((float)max(sourceWidth, 1u)), 1.0f);
    const float roughness =
        saturate(log2((float)max(sourceWidth / max(outputWidth, 1u), 1u)) / maxMip);
    const float3 normal =
        directionForCubeTexel(dispatchThreadId.z, dispatchThreadId.xy, outputWidth);
    const float3 view = normal;

    const uint sampleCount = 1024u;
    float3 prefiltered = 0.0f;
    float totalWeight = 0.0f;

    for (uint i = 0u; i < sampleCount; ++i)
    {
        const float3 halfVector =
            importanceSampleGGX(hammersley(i, sampleCount), roughness, normal);
        const float3 light = normalize(2.0f * dot(view, halfVector) * halfVector - view);
        const float nDotL = max(dot(normal, light), 0.0f);
        if (nDotL > 0.0f)
        {
            prefiltered +=
                gEnvironmentTexture.SampleLevel(
                    gEnvironmentSampler,
                    light,
                    0.0f).rgb * nDotL;
            totalWeight += nDotL;
        }
    }

    prefiltered = totalWeight > 0.0f ? prefiltered / totalWeight : 0.0f;
    gOutputPrefilter[dispatchThreadId] = float4(max(prefiltered, 0.0f), 1.0f);
}
