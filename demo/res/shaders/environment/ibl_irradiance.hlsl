#include "ibl_common.hlsli"

TextureCube<float4> gEnvironmentTexture : register(t0, space0);
SamplerState gEnvironmentSampler : register(s2, space0);
RWTexture2DArray<float4> gOutputIrradiance : register(u1, space0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint width = 0;
    uint height = 0;
    uint layers = 0;
    gOutputIrradiance.GetDimensions(width, height, layers);

    if (dispatchThreadId.x >= width ||
        dispatchThreadId.y >= height ||
        dispatchThreadId.z >= min(layers, 6u))
    {
        return;
    }

    const float3 normal =
        directionForCubeTexel(dispatchThreadId.z, dispatchThreadId.xy, width);
    const uint sampleCount = 1024u;
    float3 irradiance = 0.0f;
    float totalWeight = 0.0f;

    for (uint i = 0u; i < sampleCount; ++i)
    {
        const float2 xi = hammersley(i, sampleCount);
        const float phi = 2.0f * IC_PI * xi.x;
        const float cosTheta = sqrt(1.0f - xi.y);
        const float sinTheta = sqrt(xi.y);
        const float3 sampleDirection =
            tangentToWorld(
                float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta),
                normal);
        const float nDotL = max(dot(normal, sampleDirection), 0.0f);
        irradiance +=
            gEnvironmentTexture.SampleLevel(
                gEnvironmentSampler,
                sampleDirection,
                0.0f).rgb * nDotL;
        totalWeight += nDotL;
    }

    irradiance = totalWeight > 0.0f ? irradiance / totalWeight : 0.0f;
    gOutputIrradiance[dispatchThreadId] = float4(max(irradiance, 0.0f), 1.0f);
}
