#ifndef IC_BRDF_HLSLI
#define IC_BRDF_HLSLI

float3 lambertDiffuse(float3 baseColor)
{
    return baseColor * 0.31830988618f;
}

float distributionGGX(float3 normal, float3 halfVector, float roughness)
{
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float nDotH = saturate(dot(normal, halfVector));
    const float nDotH2 = nDotH * nDotH;
    const float denom = nDotH2 * (a2 - 1.0f) + 1.0f;
    return a2 / max(3.14159265359f * denom * denom, 1.0e-5f);
}

float geometrySchlickGGX(float nDotV, float roughness)
{
    const float r = roughness + 1.0f;
    const float k = (r * r) * 0.125f;
    return nDotV / max(nDotV * (1.0f - k) + k, 1.0e-5f);
}

float geometrySmith(float3 normal, float3 view, float3 light, float roughness)
{
    return geometrySchlickGGX(saturate(dot(normal, view)), roughness) *
        geometrySchlickGGX(saturate(dot(normal, light)), roughness);
}

float3 fresnelSchlick(float cosTheta, float3 f0)
{
    return f0 + (1.0f - f0) * pow(saturate(1.0f - cosTheta), 5.0f);
}

float3 fresnelSchlickRoughness(float cosTheta, float3 f0, float roughness)
{
    return f0 +
        (max(float3(1.0f - roughness, 1.0f - roughness, 1.0f - roughness), f0) - f0) *
        pow(saturate(1.0f - cosTheta), 5.0f);
}

// Convert normal variation inside the pixel footprint into a matching GGX
// roughness. Screen-space derivatives catch subpixel geometric detail, while
// the length of a filtered tangent-space normal retains variance that was
// already averaged away by normal-map mip filtering.
float specularAntiAliasedRoughness(
    float roughness,
    float3 shadingNormal,
    float tangentNormalLength)
{
    const float3 normalDx = ddx(shadingNormal);
    const float3 normalDy = ddy(shadingNormal);
    const float geometricVariance =
        dot(normalDx, normalDx) + dot(normalDy, normalDy);
    const float geometricKernel = min(2.0f * geometricVariance, 0.18f);
    const float normalMapKernel = min(
        saturate(1.0f - tangentNormalLength),
        0.25f);

    return sqrt(saturate(
        roughness * roughness + geometricKernel + normalMapKernel));
}

float3 pbrDirectLighting(
    float3 baseColor,
    float metallic,
    float roughness,
    float3 normal,
    float3 view,
    float3 light,
    float3 radiance)
{
    const float3 halfVector = safeNormalize(view + light, normal);
    const float nDotL = saturate(dot(normal, light));
    const float nDotV = max(saturate(dot(normal, view)), 1.0e-4f);
    const float3 f0 = lerp(float3(0.04f, 0.04f, 0.04f), baseColor, metallic);
    const float ndf = distributionGGX(normal, halfVector, roughness);
    const float g = geometrySmith(normal, view, light, roughness);
    const float3 f = fresnelSchlick(saturate(dot(halfVector, view)), f0);
    const float3 specular = (ndf * g * f) / max(4.0f * nDotV * nDotL, 1.0e-4f);
    const float3 diffuse = (1.0f - f) * (1.0f - metallic) * lambertDiffuse(baseColor);
    return (diffuse + specular) * radiance * nDotL;
}

#endif
