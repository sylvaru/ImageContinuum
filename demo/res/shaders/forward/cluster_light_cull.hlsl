#include "../common/ic_shader_common.hlsli"
#include "../common/ic_camera.hlsli"
#define IC_CLUSTERED_COMPUTE 1
#include "../common/clustered_forward.hlsli"

[numthreads(64, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint clusterIndex = dispatchThreadId.x;
    if (clusterIndex >= gFrame.clusterDimensions.w)
    {
        return;
    }

    const uint maxLightsPerCluster = max(gFrame.clusterConfig.z, 1u);
    const uint offset = clusterIndex * maxLightsPerCluster;
    const ClusterBounds clusterBounds = gClusterBounds[clusterIndex];
    const float3 clusterMin = clusterBounds.minBounds.xyz;
    const float3 clusterMax = clusterBounds.maxBounds.xyz;

    ClusterLightGrid grid;
    grid.offset = offset;
    grid.count = 0u;
    grid.pad0 = 0u;
    grid.pad1 = 0u;

    const uint visibleLightCount =
        min(gFrame.pointLightCount, IC_CLUSTERED_MAX_VISIBLE_LIGHTS);
    for (uint lightIndex = 0; lightIndex < visibleLightCount; ++lightIndex)
    {
        const float4 lightPositionRange = gVisibleLights[lightIndex].positionRange;
        const float3 lightView =
            mul(gFrame.view, float4(lightPositionRange.xyz, 1.0f)).xyz;
        const float viewDepth = -lightView.z;
        const float radius = lightPositionRange.w;

        if (viewDepth + radius <= 0.1f)
        {
            continue;
        }

        const float3 closestPoint = clamp(lightView, clusterMin, clusterMax);
        const float3 delta = lightView - closestPoint;
        const bool overlapsCluster = dot(delta, delta) <= radius * radius;

        if (overlapsCluster)
        {
            gClusterLightIndices[offset + grid.count] = lightIndex;
            grid.count++;
            if (grid.count >= maxLightsPerCluster)
            {
                break;
            }
        }
    }

    gClusterLightGrid[clusterIndex] = grid;
}
