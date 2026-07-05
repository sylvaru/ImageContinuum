#include "../common/scene_bindings.hlsli"

cbuffer gSkyboxConstants : register(b0, space0)
{
    SkyboxConstants gSkybox;
};

TextureCube<float4> gSkyboxTexture : register(t0, space0);
SamplerState gSkyboxSampler : register(s0, space0);

struct VSOut
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

VSOut VSMain(uint vertexId : SV_VertexID)
{
    const float2 positions[3] =
    {
        float2(-1.0f, -1.0f),
        float2(-1.0f, 3.0f),
        float2(3.0f, -1.0f)
    };

    VSOut output;
    output.position = float4(positions[vertexId], 1.0f, 1.0f);
    output.uv = positions[vertexId] * 0.5f + 0.5f;
    return output;
}

float4 PSMain(VSOut input) : SV_Target0
{
    const float2 ndc = input.uv * 2.0f - 1.0f;
    float verticalNdc = ndc.y;
#if defined(IC_TARGET_VULKAN)
    verticalNdc = -verticalNdc;
#endif

    const float3 direction =
        normalize(
            gSkybox.cameraForwardAndAspect.xyz +
            gSkybox.cameraRightAndNear.xyz * ndc.x * gSkybox.cameraForwardAndAspect.w * gSkybox.cameraPositionAndTanHalfFov.w -
            gSkybox.cameraUpAndFar.xyz * -verticalNdc * gSkybox.cameraPositionAndTanHalfFov.w);

    float3 color = max(gSkyboxTexture.SampleLevel(gSkyboxSampler, direction, 0.0f).rgb, 0.0f) * gSkybox.intensity;
    color *= gSkybox.exposure;
    color = color / (color + float3(1.0f, 1.0f, 1.0f));
    return float4(color, 1.0f);
}
