#ifndef IC_RAY_COMMON_HLSLI
#define IC_RAY_COMMON_HLSLI

struct Ray
{
    float3 origin;
    float3 direction;
};

float3 safeNormalize(float3 value)
{
    const float lengthSq = max(dot(value, value), 1.0e-8f);
    return value * rsqrt(lengthSq);
}

float3 skyColor(float3 direction)
{
    const float t = saturate(direction.y * 0.5f + 0.5f);
    return lerp(float3(0.04f, 0.05f, 0.07f), float3(0.55f, 0.72f, 1.0f), t);
}

#endif
