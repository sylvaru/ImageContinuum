#include "../common/ic_shader_common.hlsli"
#include "../common/ic_camera.hlsli"

struct InstanceBounds
{
    float4 centerRadius;
};

#if defined(IC_TARGET_VULKAN)
[[vk::binding(22, 0)]]
#endif
StructuredBuffer<InstanceBounds> gInstanceBounds : register(t22, space0);

#if defined(IC_TARGET_VULKAN)
[[vk::binding(29, 0)]]
#endif
Texture2D<float> gCurrentHiZ : register(t29, space0);

#if defined(IC_TARGET_VULKAN)
[[vk::binding(30, 0)]]
#endif
RWStructuredBuffer<uint> gCullStats : register(u30, space0);

#if defined(IC_TARGET_VULKAN)
[[vk::binding(31, 0)]]
#endif
StructuredBuffer<uint> gCullClassification : register(t31, space0);

static const uint CULL_OCCLUSION = 2u;
static const uint CULL_CONSERVATIVE = 3u;

float2 ndcToUv(float2 ndc)
{
#if defined(IC_TARGET_VULKAN)
    return ndc * 0.5f + 0.5f;
#else
    return float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
#endif
}

bool currentSphereOccluded(float3 center, float radius, out bool reliable)
{
    reliable = false;
    uint hiZWidth = 0u;
    uint hiZHeight = 0u;
    uint hiZMipCount = 0u;
    gCurrentHiZ.GetDimensions(0u, hiZWidth, hiZHeight, hiZMipCount);
    if (hiZWidth != gFrame.renderExtentAndHiZ.x ||
        hiZHeight != gFrame.renderExtentAndHiZ.y ||
        hiZMipCount != gFrame.renderExtentAndHiZ.z)
    {
        return false;
    }

    const float expandedRadius =
        max(radius, 0.0f) * max(gFrame.occlusionConfig.y, 1.0f);
    const float4 viewCenter = mul(gFrame.view, float4(center, 1.0f));
    const float viewDistance = -viewCenter.z;
    if (viewDistance - expandedRadius <= gFrame.cameraNearFar.x)
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
            float4(viewCenter.xyz + signs * expandedRadius, 1.0f));
        if (clip.w <= 0.0f)
        {
            return false;
        }
        const float2 ndc = clip.xy / clip.w;
        ndcMin = min(ndcMin, ndc);
        ndcMax = max(ndcMax, ndc);
    }

    if (ndcMin.x <= -1.0f || ndcMin.y <= -1.0f ||
        ndcMax.x >= 1.0f || ndcMax.y >= 1.0f)
    {
        return false;
    }

    const float2 extent = float2(hiZWidth, hiZHeight);
    float2 uv0 = ndcToUv(ndcMin);
    float2 uv1 = ndcToUv(ndcMax);
    float2 uvMin = min(uv0, uv1) - gFrame.occlusionConfig.z / extent;
    float2 uvMax = max(uv0, uv1) + gFrame.occlusionConfig.z / extent;
    uvMin = max(uvMin, 0.0f);
    uvMax = min(uvMax, 1.0f);

    const float2 pixelSize = max((uvMax - uvMin) * extent, 1.0f);
    const uint mip = min(
        hiZMipCount - 1u,
        (uint)ceil(log2(max(pixelSize.x, pixelSize.y))));

    uint mipWidth = 0u;
    uint mipHeight = 0u;
    uint ignored = 0u;
    gCurrentHiZ.GetDimensions(mip, mipWidth, mipHeight, ignored);
    const float2 mipExtent = float2(mipWidth, mipHeight);
    const uint2 maxTexel = uint2(mipWidth - 1u, mipHeight - 1u);
    const uint2 texelMin =
        min(maxTexel, (uint2)floor(uvMin * mipExtent));
    const uint2 texelMax =
        min(maxTexel, (uint2)floor(uvMax * mipExtent));

    const float d0 = gCurrentHiZ.Load(int3(texelMin, mip));
    const float d1 = gCurrentHiZ.Load(
        int3(uint2(texelMax.x, texelMin.y), mip));
    const float d2 = gCurrentHiZ.Load(
        int3(uint2(texelMin.x, texelMax.y), mip));
    const float d3 = gCurrentHiZ.Load(int3(texelMax, mip));
    const float nearestViewZ = viewCenter.z + expandedRadius;
    const float4 nearestClip = mul(
        gFrame.projection, float4(0.0f, 0.0f, nearestViewZ, 1.0f));
    const float nearestDepth = nearestClip.z / nearestClip.w;
    const float bias = max(gFrame.occlusionConfig.w, 0.0f);
    reliable = true;

    if (gFrame.renderExtentAndHiZ.w != 0u)
    {
        return nearestDepth <
            min(min(d0, d1), min(d2, d3)) - bias;
    }
    return nearestDepth >
        max(max(d0, d1), max(d2, d3)) + bias;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint instanceIndex = dispatchThreadId.x;
    if (instanceIndex >= gFrame.cullingConfig.x ||
        gFrame.occlusionDebugConfig.y == 0u)
    {
        return;
    }

    const uint classification = gCullClassification[instanceIndex];
    if (classification != CULL_OCCLUSION &&
        classification != CULL_CONSERVATIVE)
    {
        return;
    }

    bool reliable = false;
    const float4 sphere = gInstanceBounds[instanceIndex].centerRadius;
    const bool occluded =
        currentSphereOccluded(sphere.xyz, sphere.w, reliable);
    if (!reliable)
    {
        return;
    }

    if (classification == CULL_OCCLUSION && !occluded)
    {
        InterlockedAdd(gCullStats[6], 1u);
    }
    else if (classification == CULL_CONSERVATIVE && occluded)
    {
        InterlockedAdd(gCullStats[7], 1u);
    }
}
