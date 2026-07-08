#ifndef IC_CLUSTERED_FORWARD_HLSLI
#define IC_CLUSTERED_FORWARD_HLSLI

static const uint IC_CLUSTERED_MAX_VISIBLE_LIGHTS = 256;

struct ClusterBounds
{
    float4 minBounds;
    float4 maxBounds;
};

struct ClusterLightGrid
{
    uint offset;
    uint count;
    uint pad0;
    uint pad1;
};

struct VisibleLight
{
    float4 positionRange;
    float4 colorIntensity;
};

#if defined(IC_CLUSTERED_COMPUTE)

#if defined(IC_TARGET_VULKAN)
[[vk::binding(10, 0)]]
RWStructuredBuffer<ClusterBounds> gClusterBounds : register(u10, space0);
#else
RWStructuredBuffer<ClusterBounds> gClusterBounds : register(u10, space2);
#endif

#if defined(IC_TARGET_VULKAN)
[[vk::binding(11, 0)]]
RWStructuredBuffer<ClusterLightGrid> gClusterLightGrid : register(u11, space0);
#else
RWStructuredBuffer<ClusterLightGrid> gClusterLightGrid : register(u11, space2);
#endif

#if defined(IC_TARGET_VULKAN)
[[vk::binding(12, 0)]]
RWStructuredBuffer<uint> gClusterLightIndices : register(u12, space0);
#else
RWStructuredBuffer<uint> gClusterLightIndices : register(u12, space2);
#endif

#if defined(IC_TARGET_VULKAN)
[[vk::binding(13, 0)]]
StructuredBuffer<VisibleLight> gVisibleLights : register(t13, space0);
#else
StructuredBuffer<VisibleLight> gVisibleLights : register(t13, space2);
#endif

#if defined(IC_TARGET_VULKAN)
[[vk::binding(14, 0)]]
RWStructuredBuffer<uint> gClusterLightCounter : register(u14, space0);
#else
RWStructuredBuffer<uint> gClusterLightCounter : register(u14, space2);
#endif

#else

#if defined(IC_TARGET_VULKAN)
[[vk::binding(10, 0)]]
StructuredBuffer<ClusterBounds> gClusterBounds : register(t10, space0);
#else
StructuredBuffer<ClusterBounds> gClusterBounds : register(t10, space2);
#endif

#if defined(IC_TARGET_VULKAN)
[[vk::binding(11, 0)]]
StructuredBuffer<ClusterLightGrid> gClusterLightGrid : register(t11, space0);
#else
StructuredBuffer<ClusterLightGrid> gClusterLightGrid : register(t11, space2);
#endif

#if defined(IC_TARGET_VULKAN)
[[vk::binding(12, 0)]]
StructuredBuffer<uint> gClusterLightIndices : register(t12, space0);
#else
StructuredBuffer<uint> gClusterLightIndices : register(t12, space2);
#endif

#if defined(IC_TARGET_VULKAN)
[[vk::binding(13, 0)]]
StructuredBuffer<VisibleLight> gVisibleLights : register(t13, space0);
#else
StructuredBuffer<VisibleLight> gVisibleLights : register(t13, space2);
#endif

#endif

uint clusteredClusterIndexFromPixel(float4 svPosition, float3 worldPosition)
{
    const uint clusterCountX = max(gFrame.clusterDimensions.x, 1u);
    const uint clusterCountY = max(gFrame.clusterDimensions.y, 1u);
    const uint clusterCountZ = max(gFrame.clusterDimensions.z, 1u);
    const uint tileSizeX = max(gFrame.clusterConfig.x, 1u);
    const uint tileSizeY = max(gFrame.clusterConfig.y, 1u);
    
    
    const uint virtualRenderWidth =
    clusterCountX * tileSizeX;

    const uint virtualRenderHeight =
    clusterCountY * tileSizeY;

    uint pixelX = min((uint) svPosition.x, virtualRenderWidth - 1u);
    uint pixelY = min((uint) svPosition.y, virtualRenderHeight - 1u);

    // Vulkan's fragment-space Y convention is inverted relative to the
    // logical top left cluster grid used by cluster_build.hlsl.
#if defined(IC_TARGET_VULKAN)
    pixelY = virtualRenderHeight - 1u - pixelY;
#endif
    
    
    const uint x = min(pixelX / tileSizeX, clusterCountX - 1u);
    const uint y = min(pixelY / tileSizeY, clusterCountY - 1u);

    const float3 viewPosition = 
    mul(gFrame.view, float4(worldPosition, 1.0f)).xyz;
    
    const float z01 = saturate((-viewPosition.z - 0.1f) / 999.9f);
    const uint z = min((uint)(z01 * (float)clusterCountZ), clusterCountZ - 1u);

    return x + y * clusterCountX + z * clusterCountX * clusterCountY;
}

#endif
