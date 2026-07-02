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
    uint padding0;
    uint padding1;
    uint padding2;

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

#endif
