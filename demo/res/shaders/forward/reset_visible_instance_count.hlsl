#include "../common/ic_shader_common.hlsli"
#include "../common/ic_camera.hlsli"

#if defined(IC_TARGET_VULKAN)
[[vk::binding(24, 0)]]
#endif
RWStructuredBuffer<uint> gVisibleInstanceCount : register(u24, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(28, 0)]]
#endif
RWStructuredBuffer<uint> gBinCounts : register(u28, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(30, 0)]]
#endif
RWStructuredBuffer<uint> gCullStats : register(u30, space0);

[numthreads(64, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x == 0u)
    {
        gVisibleInstanceCount[0] = 0u;
    }
    if (dispatchThreadId.x < gFrame.cullingConfig.z)
    {
        gBinCounts[dispatchThreadId.x] = 0u;
    }
    if (gFrame.occlusionDebugConfig.y != 0u &&
        dispatchThreadId.x < 12u)
    {
        gCullStats[dispatchThreadId.x] = 0u;
        if (dispatchThreadId.x == 0u)
        {
            gCullStats[0] = gFrame.cullingConfig.x;
        }
        else if (dispatchThreadId.x == 8u)
        {
            gCullStats[8] = gFrame.cullingConfig.w;
        }
        else if (dispatchThreadId.x == 9u)
        {
            gCullStats[9] = gFrame.occlusionDebugConfig.z;
        }
    }
}
