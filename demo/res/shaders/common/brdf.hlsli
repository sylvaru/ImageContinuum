#ifndef IC_BRDF_HLSLI
#define IC_BRDF_HLSLI

float3 lambertDiffuse(float3 baseColor)
{
    return baseColor * 0.31830988618f;
}

#endif
