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
};

struct DrawInput
{
    DrawMetadata metadata;
    uint indexCount;
    uint firstIndex;
    int vertexOffset;
    uint commandBinOffset;
};

struct IndexedIndirectArguments
{
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

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
RWStructuredBuffer<IndexedIndirectArguments> gIndirectArguments : register(u26, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(27, 0)]]
#endif
RWStructuredBuffer<DrawMetadata> gOutputDrawMetadata : register(u27, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(28, 0)]]
#endif
RWStructuredBuffer<uint> gBinCounts : register(u28, space0);

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
    if (!sphereInsideFrustum(sphere.xyz, sphere.w))
    {
        return;
    }

    uint outputIndex = 0u;
    InterlockedAdd(gVisibleInstanceCount[0], 1u, outputIndex);
    if (outputIndex >= instanceCount)
    {
        return;
    }
    gVisibleInstances[outputIndex] = instanceIndex;

    const DrawInput input = gDrawInputs[instanceIndex];
    uint binCommandIndex = 0u;
    InterlockedAdd(gBinCounts[input.metadata.geometryBinIndex], 1u, binCommandIndex);
    if (binCommandIndex >= instanceCount)
    {
        return;
    }
    const uint commandIndex = input.commandBinOffset + binCommandIndex;

    IndexedIndirectArguments arguments;
    arguments.indexCount = input.indexCount;
    arguments.instanceCount = 1u;
    arguments.firstIndex = input.firstIndex;
    arguments.vertexOffset = input.vertexOffset;
    arguments.firstInstance = commandIndex;
    gIndirectArguments[commandIndex] = arguments;
    gOutputDrawMetadata[commandIndex] = input.metadata;
}
