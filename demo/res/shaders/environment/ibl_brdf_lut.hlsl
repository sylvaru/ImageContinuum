#include "ibl_common.hlsli"

TextureCube<float4> gUnusedTexture : register(t0, space0);
SamplerState gUnusedSampler : register(s2, space0);
RWTexture2D<float4> gOutputBrdfLut : register(u1, space0);

float2 integrateBRDF(float nDotV, float roughness)
{
    const float3 view =
        float3(sqrt(max(1.0f - nDotV * nDotV, 0.0f)), 0.0f, nDotV);
    const float3 normal = float3(0.0f, 0.0f, 1.0f);

    float a = 0.0f;
    float b = 0.0f;
    const uint sampleCount = 1024u;

    for (uint i = 0u; i < sampleCount; ++i)
    {
        const float3 halfVector =
            importanceSampleGGX(hammersley(i, sampleCount), roughness, normal);
        const float3 light =
            normalize(2.0f * dot(view, halfVector) * halfVector - view);

        const float nDotL = max(light.z, 0.0f);
        const float nDotH = max(halfVector.z, 0.0f);
        const float vDotH = max(dot(view, halfVector), 0.0f);

        if (nDotL > 0.0f)
        {
            const float g = geometrySmith(normal, view, light, roughness);
            const float gVis = (g * vDotH) / max(nDotH * nDotV, 0.0001f);
            const float fc = pow(1.0f - vDotH, 5.0f);
            a += (1.0f - fc) * gVis;
            b += fc * gVis;
        }
    }

    return float2(a, b) / float(sampleCount);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint width = 0;
    uint height = 0;
    gOutputBrdfLut.GetDimensions(width, height);

    if (dispatchThreadId.x >= width || dispatchThreadId.y >= height)
    {
        return;
    }

    const float2 uv = (float2(dispatchThreadId.xy) + 0.5f) / float2(width, height);
    gOutputBrdfLut[dispatchThreadId.xy] =
        float4(integrateBRDF(max(uv.x, 0.001f), uv.y), 0.0f, 1.0f);
}
