#include "../common/ic_shader_common.hlsli"
#include "../common/ic_camera.hlsli"

struct InstanceBounds
{
    float4 centerRadius;
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

struct DrawInput
{
    DrawMetadata metadata;
    uint indexCount;
    uint firstIndex;
    int vertexOffset;
    uint commandBinOffset;
    uint commandBinCapacity;
    uint commandBinLocalIndex;
    uint padding1;
    uint padding2;
};

struct IndexedIndirectArguments
{
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

#if defined(IC_TARGET_DX12)
struct IndexedIndirectCommand
{
    uint4 drawConstants;
    IndexedIndirectArguments draw;
};
#else
typedef IndexedIndirectArguments IndexedIndirectCommand;
#endif

#if defined(IC_TARGET_VULKAN)
[[vk::binding(22, 0)]]
#endif
StructuredBuffer<InstanceBounds> gInstanceBounds : register(t22, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(25, 0)]]
#endif
StructuredBuffer<DrawInput> gDrawInputs : register(t25, space0);

#if defined(IC_TARGET_VULKAN)
[[vk::binding(23, 0)]]
#endif
RWStructuredBuffer<uint> gVisibleInstances : register(u23, space0);

#if defined(IC_TARGET_VULKAN)
[[vk::binding(24, 0)]]
#endif
RWStructuredBuffer<uint> gVisibleInstanceCount : register(u24, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(26, 0)]]
#endif
RWStructuredBuffer<IndexedIndirectCommand> gIndirectArguments : register(u26, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(27, 0)]]
RWStructuredBuffer<DrawMetadata> gOutputDrawMetadata : register(u27, space0);
#endif
#if defined(IC_TARGET_VULKAN)
[[vk::binding(28, 0)]]
#endif
RWStructuredBuffer<uint> gBinCounts : register(u28, space0);

#if defined(IC_TARGET_VULKAN)
[[vk::binding(29, 0)]]
#endif
Texture2D<float> gPreviousHiZ : register(t29, space0);

#if defined(IC_TARGET_VULKAN)
[[vk::binding(30, 0)]]
#endif
RWStructuredBuffer<uint> gCullStats : register(u30, space0);

#if defined(IC_TARGET_VULKAN)
[[vk::binding(31, 0)]]
#endif
RWStructuredBuffer<uint> gCullClassification : register(u31, space0);

static const uint CULL_VISIBLE = 0u;
static const uint CULL_FRUSTUM = 1u;
static const uint CULL_OCCLUSION = 2u;
static const uint CULL_CONSERVATIVE = 3u;

bool sphereInsideFrustum(float3 center, float radius)
{
    const float4 viewCenter = mul(gFrame.view, float4(center, 1.0f));
    const float4 clip = mul(gFrame.projection, viewCenter);
    const float w = clip.w;
    const float safetyScale = 1.1f;

    // Each side plane includes clip.w. Its support radius must therefore
    // include both the projected axis and the perspective w contribution.
    const float horizontalSupport = safetyScale * radius *
        sqrt(gFrame.projection[0][0] * gFrame.projection[0][0] + 1.0f);
    const float verticalSupport = safetyScale * radius *
        sqrt(gFrame.projection[1][1] * gFrame.projection[1][1] + 1.0f);

    // Keep spheres intersecting the camera plane. They are difficult to
    // classify conservatively with a perspective projection and must render.
    if (w < -radius)
    {
        return false;
    }

    if (clip.x + w < -horizontalSupport ||
        w - clip.x < -horizontalSupport)
    {
        return false;
    }
    if (clip.y + w < -verticalSupport ||
        w - clip.y < -verticalSupport)
    {
        return false;
    }

    const float viewDistance = -viewCenter.z;
    if (viewDistance + safetyScale * radius < gFrame.cameraNearFar.x ||
        viewDistance - safetyScale * radius > gFrame.cameraNearFar.y)
    {
        return false;
    }

    return true;
}

bool reserveBounded(
    RWStructuredBuffer<uint> counter,
    uint counterIndex,
    uint capacity,
    out uint reservedIndex)
{
    uint observed = counter[counterIndex];
    [loop]
    while (observed < capacity)
    {
        uint original = 0u;
        InterlockedCompareExchange(
            counter[counterIndex], observed, observed + 1u, original);
        if (original == observed)
        {
            reservedIndex = observed;
            return true;
        }
        observed = original;
    }

    reservedIndex = capacity;
    return false;
}

float2 ndcToUv(float2 ndc)
{
#if defined(IC_TARGET_VULKAN)
    return ndc * 0.5f + 0.5f;
#else
    return float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
#endif
}

bool sphereOccluded(float3 center, float radius, out bool reliable)
{
    reliable = false;
    if (gFrame.cullingConfig.w == 0u ||
        gFrame.renderExtentAndHiZ.x == 0u ||
        gFrame.renderExtentAndHiZ.y == 0u ||
        gFrame.renderExtentAndHiZ.z == 0u)
    {
        return false;
    }

    uint hiZWidth = 0u;
    uint hiZHeight = 0u;
    uint hiZMipCount = 0u;
    gPreviousHiZ.GetDimensions(0u, hiZWidth, hiZHeight, hiZMipCount);
    if (hiZWidth != gFrame.renderExtentAndHiZ.x ||
        hiZHeight != gFrame.renderExtentAndHiZ.y ||
        hiZMipCount != gFrame.renderExtentAndHiZ.z)
    {
        return false;
    }

    const float expandedRadius =
        max(radius, 0.0f) * max(gFrame.occlusionConfig.y, 1.0f);
    const float4 previousViewCenter =
        mul(gFrame.previousView, float4(center, 1.0f));
    const float viewDistance = -previousViewCenter.z;

    // Anything touching the previous camera's near plane can produce an
    // unbounded projection. Rendering it is the only conservative choice.
    if (viewDistance - expandedRadius <= gFrame.occlusionConfig.x)
    {
        return false;
    }

    float2 ndcMin = float2(1.0e30f, 1.0e30f);
    float2 ndcMax = float2(-1.0e30f, -1.0e30f);
    [unroll]
    for (uint corner = 0u; corner < 8u; ++corner)
    {
        const float3 signs = float3(
            (corner & 1u) != 0u ? 1.0f : -1.0f,
            (corner & 2u) != 0u ? 1.0f : -1.0f,
            (corner & 4u) != 0u ? 1.0f : -1.0f);
        const float4 clip = mul(
            gFrame.projection,
            float4(previousViewCenter.xyz + signs * expandedRadius, 1.0f));
        if (clip.w <= 0.0f)
        {
            return false;
        }
        const float2 ndc = clip.xy / clip.w;
        ndcMin = min(ndcMin, ndc);
        ndcMax = max(ndcMax, ndc);
    }

    // Newly entering or edge-clipped bounds cannot be classified from the
    // previous image without risking false occlusion.
    if (ndcMin.x <= -1.0f || ndcMin.y <= -1.0f ||
        ndcMax.x >= 1.0f || ndcMax.y >= 1.0f)
    {
        return false;
    }

    float2 uv0 = ndcToUv(ndcMin);
    float2 uv1 = ndcToUv(ndcMax);
    float2 uvMin = min(uv0, uv1);
    float2 uvMax = max(uv0, uv1);

    const float2 extent = float2(hiZWidth, hiZHeight);
    const float2 pixelExpansion =
        gFrame.occlusionConfig.z / extent;
    uvMin = max(float2(0.0f, 0.0f), uvMin - pixelExpansion);
    uvMax = min(float2(1.0f, 1.0f), uvMax + pixelExpansion);

    const float2 pixelSize = max((uvMax - uvMin) * extent, 1.0f);
    const float maxPixelSpan = max(pixelSize.x, pixelSize.y);
    const uint mip = min(
        hiZMipCount - 1u,
        (uint)ceil(log2(maxPixelSpan)));

    uint mipWidth = 0u;
    uint mipHeight = 0u;
    uint ignoredMipCount = 0u;
    gPreviousHiZ.GetDimensions(
        mip, mipWidth, mipHeight, ignoredMipCount);
    const float2 mipExtent = float2(mipWidth, mipHeight);
    const uint2 texelMin = min(
        uint2(mipWidth - 1u, mipHeight - 1u),
        (uint2)floor(uvMin * mipExtent));
    const uint2 texelMax = min(
        uint2(mipWidth - 1u, mipHeight - 1u),
        (uint2)floor(uvMax * mipExtent));

    const float d0 = gPreviousHiZ.Load(int3(texelMin, mip));
    const float d1 = gPreviousHiZ.Load(
        int3(uint2(texelMax.x, texelMin.y), mip));
    const float d2 = gPreviousHiZ.Load(
        int3(uint2(texelMin.x, texelMax.y), mip));
    const float d3 = gPreviousHiZ.Load(int3(texelMax, mip));

    const float nearestViewZ = previousViewCenter.z + expandedRadius;
    const float4 nearestClip = mul(
        gFrame.projection, float4(0.0f, 0.0f, nearestViewZ, 1.0f));
    const float nearestDepth = nearestClip.z / nearestClip.w;
    const bool reversedZ = gFrame.renderExtentAndHiZ.w != 0u;
    const float depthBias = max(gFrame.occlusionConfig.w, 0.0f);

    if (reversedZ)
    {
        const float farthestDepth = min(min(d0, d1), min(d2, d3));
        reliable = true;
        return nearestDepth < farthestDepth - depthBias;
    }

    const float farthestDepth = max(max(d0, d1), max(d2, d3));
    reliable = true;
    return nearestDepth > farthestDepth + depthBias;
}

void incrementStat(uint index)
{
    if (gFrame.occlusionDebugConfig.y != 0u)
    {
        InterlockedAdd(gCullStats[index], 1u);
    }
}

[numthreads(64, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint instanceIndex = dispatchThreadId.x;
    const uint instanceCount = gFrame.cullingConfig.x;

    if (instanceIndex >= instanceCount)
    {
        return;
    }

    const float4 sphere = gInstanceBounds[instanceIndex].centerRadius;
    uint cullState = CULL_VISIBLE;
    if (!sphereInsideFrustum(sphere.xyz, sphere.w))
    {
        cullState = CULL_FRUSTUM;
        incrementStat(1u);
    }
    else if (gFrame.occlusionDebugConfig.z != 0u)
    {
        bool occlusionReliable = false;
        if (sphereOccluded(
                sphere.xyz, sphere.w, occlusionReliable))
        {
            cullState = CULL_OCCLUSION;
            incrementStat(2u);
        }
        else if (!occlusionReliable)
        {
            cullState = CULL_CONSERVATIVE;
            incrementStat(3u);
        }
        else
        {
            incrementStat(4u);
        }
    }
    else
    {
        incrementStat(4u);
    }
    if (gFrame.occlusionDebugConfig.y != 0u)
    {
        gCullClassification[instanceIndex] = cullState;
    }

    const bool classificationView =
        gFrame.occlusionDebugConfig.x == 2u;
    if (!classificationView &&
        (cullState == CULL_FRUSTUM || cullState == CULL_OCCLUSION))
    {
        return;
    }

    const DrawInput input = gDrawInputs[instanceIndex];
    if (input.metadata.geometryBinIndex >= gFrame.cullingConfig.z ||
        input.commandBinCapacity == 0u ||
        input.commandBinOffset >= instanceCount ||
        input.commandBinLocalIndex >= input.commandBinCapacity)
    {
        incrementStat(5u);
        return;
    }

    uint outputIndex = 0u;
    if (!classificationView &&
        !reserveBounded(
            gVisibleInstanceCount, 0u, instanceCount, outputIndex))
    {
        incrementStat(5u);
        return;
    }
    if (!classificationView)
    {
        gVisibleInstances[outputIndex] = instanceIndex;
    }

    uint binCommandIndex = 0u;
    if (classificationView)
    {
        binCommandIndex = input.commandBinLocalIndex;
        InterlockedMax(
            gBinCounts[input.metadata.geometryBinIndex],
            min(input.commandBinCapacity, binCommandIndex + 1u));
    }
    else if (!reserveBounded(
                 gBinCounts,
                 input.metadata.geometryBinIndex,
                 input.commandBinCapacity,
                 binCommandIndex))
    {
        incrementStat(5u);
        return;
    }
    const uint commandIndex = input.commandBinOffset + binCommandIndex;
    if (commandIndex >= instanceCount)
    {
        incrementStat(5u);
        return;
    }

    IndexedIndirectArguments arguments;
    arguments.indexCount = input.indexCount;
    arguments.instanceCount = 1u;
    arguments.firstIndex = input.firstIndex;
    arguments.vertexOffset = input.vertexOffset;
    arguments.firstInstance = commandIndex;
#if defined(IC_TARGET_DX12)
    IndexedIndirectCommand command;
    command.drawConstants = uint4(
        input.metadata.transformIndex,
        input.metadata.meshIndex,
        input.metadata.materialIndex,
        cullState);
    command.draw = arguments;
    gIndirectArguments[commandIndex] = command;
#else
    gIndirectArguments[commandIndex] = arguments;
#endif
#if defined(IC_TARGET_VULKAN)
    DrawMetadata metadata = input.metadata;
    metadata.cullState = cullState;
    gOutputDrawMetadata[commandIndex] = metadata;
#endif
}
