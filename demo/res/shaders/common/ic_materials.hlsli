#ifndef IC_MATERIALS_HLSLI
#define IC_MATERIALS_HLSLI

static const uint IC_INVALID_BINDLESS_INDEX = 0xffffffffu;

struct ObjectData
{
    float4x4 world;
    float4x4 inverseTransposeWorld;
};

struct MaterialData
{
    float4 baseColorFactor;
    float4 emissiveFactor;

    float metallicFactor;
    float roughnessFactor;
    float alphaCutoff;
    float occlusionStrength;

    uint flags;
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
    uint padding0;
};

struct DrawConstants
{
    uint objectIndex;
    uint meshIndex;
    uint materialIndex;
    uint flags;
};

StructuredBuffer<ObjectData> gObjects : register(t0, space0);
StructuredBuffer<MaterialData> gMaterials : register(t1, space0);

#if defined(IC_TARGET_VULKAN)
[[vk::push_constant]]
#endif
ConstantBuffer<DrawConstants> gDraw : register(b1, space0);

#endif
