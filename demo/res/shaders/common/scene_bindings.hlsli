#ifndef IC_SCENE_BINDINGS_HLSLI
#define IC_SCENE_BINDINGS_HLSLI

static const uint IC_MAX_PATH_TRACE_POINT_LIGHTS = 8;

struct PathTraceConstants
{
    uint2 renderSize;
    uint frameIndex;
    uint accumulatedSampleCount;

    float exposure;
    uint resetAccumulation;
    uint maxBounces;
    uint samplesPerPixel;

    uint sceneVertexCount;
    uint sceneMaterialCount;
    uint sceneTriangleCount;
    uint sceneBvhNodeCount;

    uint useSceneGeometry;
    uint environmentEnabled;
    uint sceneEmissiveTriangleIndex;
    uint referenceMode;

    float environmentIntensity;
    float environmentExposure;
    uint sceneEmissiveTriangleCount;
    uint paddingScene1;

    float4 cameraPositionAndTanHalfFov;
    float4 cameraForwardAndAspect;
    float4 cameraRightAndNear;
    float4 cameraUpAndFar;

    uint pointLightCount;
    float3 padding0;

    float4 pointLightPositionRange[IC_MAX_PATH_TRACE_POINT_LIGHTS];
    float4 pointLightColorIntensity[IC_MAX_PATH_TRACE_POINT_LIGHTS];

    float4 directionalLightDirectionIntensity;
    float4 directionalLightColor;
};

struct TonemapConstants
{
    uint2 renderSize;
    float exposure;
    uint padding0;
};

struct SkyboxConstants
{
    float4 cameraPositionAndTanHalfFov;
    float4 cameraForwardAndAspect;
    float4 cameraRightAndNear;
    float4 cameraUpAndFar;
    float intensity;
    float exposure;
    float2 padding0;
};

#endif
