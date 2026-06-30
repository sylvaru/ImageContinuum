#ifndef IC_SHADER_COMMON_HLSLI
#define IC_SHADER_COMMON_HLSLI

float3 safeNormalize(float3 value, float3 fallback)
{
    const float lenSq = dot(value, value);
    return lenSq > 0.000001f ? value * rsqrt(lenSq) : fallback;
}

#endif
