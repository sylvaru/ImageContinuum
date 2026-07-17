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

struct DrawMetadata
{
    uint meshIndex;
    uint materialIndex;
    uint transformIndex;
    uint instanceIndex;
    uint geometryRangeIndex;
    uint pipelineBinIndex;
    uint materialBinIndex;
    uint geometryBinIndex;
    uint cullState;
    uint padding0;
    uint padding1;
    uint padding2;
};

StructuredBuffer<ObjectData> gObjects : register(t0, space0);
StructuredBuffer<MaterialData> gMaterials : register(t1, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(25, 0)]]
StructuredBuffer<DrawMetadata> gDrawMetadata : register(t25, space0);
#endif

#if defined(IC_TARGET_VULKAN)
[[vk::push_constant]]
#endif
ConstantBuffer<DrawConstants> gDraw : register(b1, space0);

DrawMetadata resolveDrawMetadata(uint instanceId)
{
#if defined(IC_TARGET_VULKAN)
    if (gFrame.cullingConfig.y != 0u)
    {
        return gDrawMetadata[instanceId];
    }
#else
    (void)instanceId;
#endif
    DrawMetadata result = (DrawMetadata)0;
    result.meshIndex = gDraw.meshIndex;
    result.materialIndex = gDraw.materialIndex;
    result.transformIndex = gDraw.objectIndex;
    result.instanceIndex = gDraw.objectIndex;
    result.cullState = gDraw.flags;
    return result;
}

float4 cullDebugColor(uint cullState)
{
    if (cullState == 1u)
    {
        return float4(0.15f, 0.35f, 1.0f, 1.0f);
    }
    if (cullState == 2u)
    {
        return float4(1.0f, 0.12f, 0.08f, 1.0f);
    }
    if (cullState == 3u)
    {
        return float4(1.0f, 0.78f, 0.05f, 1.0f);
    }
    return float4(0.1f, 0.95f, 0.25f, 1.0f);
}

float4 cullDebugPosition(float4 clipPosition, uint cullState)
{
    // Reveal rejected silhouettes through their occluders in the debug-only
    // classification view. Normal rendering keeps the original depth.
    if (gFrame.occlusionDebugConfig.x == 2u &&
        (cullState == 1u || cullState == 2u))
    {
        clipPosition.z =
            gFrame.renderExtentAndHiZ.w != 0u
                ? clipPosition.w
                : 0.0f;
    }
    return clipPosition;
}

#endif
