#ifndef IC_BVH_HLSLI
#define IC_BVH_HLSLI

struct PathTraceBVHNode
{
    float3 boundsMin;
    uint leftFirst;
    float3 boundsMax;
    uint count;
};

struct PathTraceVertex
{
    float4 position;
    float4 normal;
    float4 tangent;
    float4 texCoord;
};

struct PathTraceMaterial
{
    float4 baseColor;
    float4 emissive;

    float metallicFactor;
    float roughnessFactor;
    float occlusionStrength;
    uint materialType;

    uint baseColorTextureIndex;
    uint normalTextureIndex;
    uint metallicRoughnessTextureIndex;
    uint occlusionTextureIndex;

    uint emissiveTextureIndex;
    uint baseColorSamplerIndex;
    uint normalSamplerIndex;
    uint metallicRoughnessSamplerIndex;

    uint occlusionSamplerIndex;
    uint emissiveSamplerIndex;
    float2 padding0;
};

struct PathTraceTriangle
{
    uint i0;
    uint i1;
    uint i2;
    uint materialIndex;
};

#endif
