#include "../common/ic_shader_common.hlsli"
#include "../common/ic_camera.hlsli"

#if defined(IC_TARGET_VULKAN)
[[vk::binding(20, 0)]]
#endif
Texture2D<float> gHiZSource : register(t20, space0);

#if defined(IC_TARGET_VULKAN)
[[vk::binding(21, 0)]]
#endif
RWTexture2D<float> gHiZOutput : register(u21, space0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint outputWidth = 0;
    uint outputHeight = 0;
    gHiZOutput.GetDimensions(outputWidth, outputHeight);

    const uint2 dst = dispatchThreadId.xy;
    if (dst.x >= outputWidth || dst.y >= outputHeight)
    {
        return;
    }

    uint sourceWidth = 0;
    uint sourceHeight = 0;
    gHiZSource.GetDimensions(sourceWidth, sourceHeight);

    const bool sameSize = sourceWidth == outputWidth && sourceHeight == outputHeight;
    if (sameSize)
    {
        gHiZOutput[dst] = gHiZSource.Load(int3(dst, 0));
        return;
    }

    const uint2 src0 = sameSize ? dst : dst * 2u;
    const uint2 src1 = min(src0 + uint2(1u, 0u), uint2(sourceWidth - 1u, sourceHeight - 1u));
    const uint2 src2 = min(src0 + uint2(0u, 1u), uint2(sourceWidth - 1u, sourceHeight - 1u));
    const uint2 src3 = min(src0 + uint2(1u, 1u), uint2(sourceWidth - 1u, sourceHeight - 1u));

    const float d0 = gHiZSource.Load(int3(src0, 0));
    const float d1 = gHiZSource.Load(int3(src1, 0));
    const float d2 = gHiZSource.Load(int3(src2, 0));
    const float d3 = gHiZSource.Load(int3(src3, 0));

    const bool reversedZ = gFrame.renderExtentAndHiZ.w != 0u;
    // Occlusion needs the farthest depth represented by the tile. Rejecting
    // against the nearest depth would falsely hide bounds when even one pixel
    // in the tile contains more distant geometry (or clear depth).
    const float reducedDepth =
        reversedZ
            ? min(min(d0, d1), min(d2, d3))
            : max(max(d0, d1), max(d2, d3));

    gHiZOutput[dst] = reducedDepth;
}
