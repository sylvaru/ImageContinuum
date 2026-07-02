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
};

struct PathTraceMaterial
{
    float4 baseColor;
    float4 emissive;
    uint baseColorTextureIndex;
    uint normalTextureIndex;
    uint materialType;
    float roughness;
    float metallic;
    float padding;
    float2 padding1;
};

struct PathTraceTriangle
{
    uint i0;
    uint i1;
    uint i2;
    uint materialIndex;
};

#endif
