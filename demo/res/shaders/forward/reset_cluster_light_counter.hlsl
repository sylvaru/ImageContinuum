#include "../common/ic_shader_common.hlsli"
#include "../common/ic_camera.hlsli"
#define IC_CLUSTERED_COMPUTE 1
#include "../common/clustered_forward.hlsli"

[numthreads(1, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gClusterLightCounter[0] = 0u;
}
