#include "../common/scene_bindings.hlsli"

cbuffer gTonemapConstants : register(b0, space0)
{
    TonemapConstants gConstants;
};

Texture2D<float4> gHDR : register(t2, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::image_format("rgba8")]]
#endif
RWTexture2D<float4> gBackBuffer : register(u1, space0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gConstants.renderSize))
    {
        return;
    }

    float3 color = gHDR[pixel].rgb * gConstants.exposure;
    color = color / (color + 1.0f);
    color = pow(max(color, 0.0f), 1.0f / 2.2f);

#if defined(IC_TARGET_VULKAN)
    gBackBuffer[pixel] = float4(color.bgr, 1.0f);
#else
    gBackBuffer[pixel] = float4(color, 1.0f);
#endif
}
