#ifndef IC_CAMERA_HLSLI
#define IC_CAMERA_HLSLI

static const uint IC_MAX_POINT_LIGHTS = 8;

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

    uint environmentEnabled;
    uint prefilteredMipCount;
    float environmentIntensity;
    float environmentExposure;

    uint pointLightCount;
    float3 padding1;

    float4 pointLightPositionRange[IC_MAX_POINT_LIGHTS];
    float4 pointLightColorIntensity[IC_MAX_POINT_LIGHTS];

    uint4 clusterDimensions;
    uint4 clusterConfig;
    uint4 renderExtentAndHiZ; // x width, y height, z hi-z mip count, w reversed-Z flag.
    uint4 cullingConfig;      // x object count, y visible count/debug, z/w reserved.
    float4 cameraNearFar;     // x near, y far, z/w reserved.
};

ConstantBuffer<FrameConstants> gFrame : register(b0, space0);

#endif
