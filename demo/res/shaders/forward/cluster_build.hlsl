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

    const uint clusterCountX = max(gFrame.clusterDimensions.x, 1u);
    const uint clusterCountY = max(gFrame.clusterDimensions.y, 1u);
    const uint clusterCountZ = max(gFrame.clusterDimensions.z, 1u);
    const uint tileSizeX = max(gFrame.clusterConfig.x, 1u);
    const uint tileSizeY = max(gFrame.clusterConfig.y, 1u);
    const uint clusterX = clusterIndex % clusterCountX;
    const uint clusterY = (clusterIndex / clusterCountX) % clusterCountY;
    const uint clusterZ = clusterIndex / (clusterCountX * clusterCountY);
    const float2 viewportSize =
        float2(clusterCountX * tileSizeX, clusterCountY * tileSizeY);

    const float2 pixelMin = float2(clusterX * tileSizeX, clusterY * tileSizeY);
    const float2 pixelMax = pixelMin + float2(tileSizeX, tileSizeY);
    const float2 ndcMin =
        float2(pixelMin.x / viewportSize.x * 2.0f - 1.0f,
               1.0f - pixelMax.y / viewportSize.y * 2.0f);
    const float2 ndcMax =
        float2(pixelMax.x / viewportSize.x * 2.0f - 1.0f,
               1.0f - pixelMin.y / viewportSize.y * 2.0f);

    const float nearPlane = 0.1f;
    const float farPlane = 1000.0f;
    const float z0 =
        lerp(nearPlane, farPlane, (float)clusterZ / (float)clusterCountZ);
    const float z1 =
        lerp(nearPlane, farPlane, (float)(clusterZ + 1u) / (float)clusterCountZ);
    const float invProjX = 1.0f / max(abs(gFrame.projection[0][0]), 0.0001f);
    const float invProjY = 1.0f / max(abs(gFrame.projection[1][1]), 0.0001f);
    const float ySign = gFrame.projection[1][1] < 0.0f ? -1.0f : 1.0f;

    float3 minBounds = float3(3.402823e38f, 3.402823e38f, 3.402823e38f);
    float3 maxBounds = float3(-3.402823e38f, -3.402823e38f, -3.402823e38f);
    const float2 corners[4] =
    {
        float2(ndcMin.x, ndcMin.y),
        float2(ndcMax.x, ndcMin.y),
        float2(ndcMin.x, ndcMax.y),
        float2(ndcMax.x, ndcMax.y)
    };
    for (uint depthIndex = 0; depthIndex < 2u; ++depthIndex)
    {
        const float depth = depthIndex == 0u ? z0 : z1;
        for (uint cornerIndex = 0; cornerIndex < 4u; ++cornerIndex)
        {
            const float2 ndc = corners[cornerIndex];
            const float3 viewPosition =
                float3(
                    ndc.x * depth * invProjX,
                    ndc.y * depth * invProjY * ySign,
                    -depth);
            minBounds = min(minBounds, viewPosition);
            maxBounds = max(maxBounds, viewPosition);
        }
    }

    gClusterBounds[clusterIndex].minBounds = float4(minBounds, 0.0f);
    gClusterBounds[clusterIndex].maxBounds = float4(maxBounds, 0.0f);
}
