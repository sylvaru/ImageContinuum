#define IC_PI 3.14159265358979323846f

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

float radicalInverseVdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

float2 hammersley(uint i, uint n)
{
    return float2(float(i) / float(n), radicalInverseVdC(i));
}

float3 tangentToWorld(float3 sampleVector, float3 normal)
{
    const float3 up = abs(normal.z) < 0.999f
        ? float3(0.0f, 0.0f, 1.0f)
        : float3(1.0f, 0.0f, 0.0f);
    const float3 tangent = normalize(cross(up, normal));
    const float3 bitangent = cross(normal, tangent);
    return normalize(
        tangent * sampleVector.x +
        bitangent * sampleVector.y +
        normal * sampleVector.z);
}

float3 importanceSampleGGX(float2 xi, float roughness, float3 normal)
{
    const float a = roughness * roughness;
    const float phi = 2.0f * IC_PI * xi.x;
    const float cosTheta =
        sqrt((1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y));
    const float sinTheta = sqrt(max(1.0f - cosTheta * cosTheta, 0.0f));
    return tangentToWorld(
        float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta),
        normal);
}

float geometrySchlickGGX(float nDotV, float roughness)
{
    const float a = roughness;
    const float k = (a * a) / 2.0f;
    return nDotV / max(nDotV * (1.0f - k) + k, 0.0001f);
}

float geometrySmith(float3 normal, float3 view, float3 light, float roughness)
{
    return geometrySchlickGGX(max(dot(normal, view), 0.0f), roughness) *
        geometrySchlickGGX(max(dot(normal, light), 0.0f), roughness);
}
