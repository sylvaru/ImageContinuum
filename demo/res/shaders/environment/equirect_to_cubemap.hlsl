#define IC_PI 3.14159265358979323846f

Texture2D<float4> gEquirectTexture : register(t0, space0);
SamplerState gEquirectSampler : register(s2, space0);
RWTexture2DArray<float4> gOutputCubemap : register(u1, space0);

float3 directionForCubeTexel(uint face, uint2 texel, uint size)
{
    const float2 uv = ((float2(texel) + 0.5f) / (float)size) * 2.0f - 1.0f;
    const float u = uv.x;
    const float v = -uv.y;

    if (face == 0u) return normalize(float3(1.0f, v, -u));
    if (face == 1u) return normalize(float3(-1.0f, v, u));
    if (face == 2u) return normalize(float3(u, 1.0f, -v));
    if (face == 3u) return normalize(float3(u, -1.0f, v));
    if (face == 4u) return normalize(float3(u, v, 1.0f));
    return normalize(float3(-u, v, -1.0f));
}

float2 equirectUv(float3 direction)
{
    const float phi = atan2(direction.z, direction.x);
    const float theta = asin(clamp(direction.y, -1.0f, 1.0f));
    return float2(phi / (2.0f * IC_PI) + 0.5f, 0.5f - theta / IC_PI);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint width = 0;
    uint height = 0;
    uint layers = 0;
    gOutputCubemap.GetDimensions(width, height, layers);

    if (dispatchThreadId.x >= width ||
        dispatchThreadId.y >= height ||
        dispatchThreadId.z >= min(layers, 6u))
    {
        return;
    }

    const float3 direction =
        directionForCubeTexel(
            dispatchThreadId.z,
            dispatchThreadId.xy,
            width);
    gOutputCubemap[dispatchThreadId] =
        float4(max(gEquirectTexture.SampleLevel(gEquirectSampler, equirectUv(direction), 0.0f).rgb, 0.0f), 1.0f);
}
