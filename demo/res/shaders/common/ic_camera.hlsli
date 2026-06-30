#ifndef IC_CAMERA_HLSLI
#define IC_CAMERA_HLSLI

struct FrameConstants
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewProjection;

    float3 cameraPosition;
    float time;

    float3 lightDirection;
    float lightIntensity;

    float3 lightColor;
    float padding0;
};

ConstantBuffer<FrameConstants> gFrame : register(b0, space0);

#endif
