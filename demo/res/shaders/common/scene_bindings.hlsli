#ifndef IC_SCENE_BINDINGS_HLSLI
#define IC_SCENE_BINDINGS_HLSLI

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
    float environmentIntensity;
    float environmentExposure;

    float4 cameraPositionAndTanHalfFov;
    float4 cameraForwardAndAspect;
    float4 cameraRightAndNear;
    float4 cameraUpAndFar;
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
