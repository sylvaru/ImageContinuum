#ifndef IC_GI_RAY_QUERY_COMMON_HLSLI
#define IC_GI_RAY_QUERY_COMMON_HLSLI

static const uint GI_INVALID_INDEX = 0xffffffffu;
static const uint GI_RESERVED_INDEX = 0xfffffffeu;
static const uint GI_GEOMETRY_OPAQUE = 1u << 0u;
static const uint GI_GEOMETRY_ALPHA_TESTED = 1u << 1u;
static const uint GI_GEOMETRY_DOUBLE_SIDED = 1u << 2u;
static const uint GI_MATERIAL_DOUBLE_SIDED = 1u << 0u;
static const uint GI_MATERIAL_ALPHA_BLEND = 1u << 1u;
static const uint GI_MATERIAL_ALPHA_MASK = 1u << 2u;
static const uint GI_MAX_MODEL_BUFFERS = 256u;
static const uint GI_MAX_TEXTURES = 4096u;
static const uint GI_MAX_SAMPLERS = 256u;
static const uint GI_VERTEX_STRIDE = 72u;
static const uint GI_MATERIAL_STRIDE = 96u;
static const uint GI_GEOMETRY_STRIDE = 48u;
static const uint GI_INSTANCE_STRIDE = 160u;
static const uint GI_SURFEL_STRIDE = 128u;
static const uint GI_HIT_RECORD_STRIDE = 128u;

struct GiTraceConstants
{
    uint frameIndex;
    uint sceneGeneration;
    uint geometryCount;
    uint instanceCount;
    uint materialCount;
    uint textureCount;
    uint samplerCount;
    uint maxUpdates;
    uint rayBudget;
    uint raysPerSurfel;
    uint debugView;
    uint environmentEnabled;
    float environmentIntensity;
    float rayMinDistance;
    float rayMaxDistance;
    uint maxSurfels;
    float cellSize;
    float invCellSize;
    uint hashBucketCount;
    uint candidatesPerCell;
    float gatherRadiusScale;
    float normalThreshold;
    float planeThreshold;
    uint maxSurfelAge;
    uint reducedWidth;
    uint reducedHeight;
    uint evaluationDivisor;
    uint feedbackEnabled;
    float confidenceBlend;
    uint emissiveInstanceIndex;
    uint giFlags;
    uint freezeAfterFrame;
};

#if defined(IC_TARGET_VULKAN)
[[vk::push_constant]]
#endif
ConstantBuffer<GiTraceConstants> gGiTraceConstants : register(b0, space0);

// Shared frame constants (camera + scene lights). Binds the same GpuFrameData
// buffer clustered forward uses; the struct layout must stay in lock-step with
// FrameConstants in ic_camera.hlsli / GpuFrameData in renderer_gpu_assets.h.
static const uint GI_MAX_POINT_LIGHTS = 8u;
struct GiFrameConstants
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewProjection;
    float4x4 previousView;
    float4x4 previousViewProjection;
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
    float environmentTransportExposure;
    float2 padding1;
    float4 pointLightPositionRange[GI_MAX_POINT_LIGHTS];
    float4 pointLightColorIntensity[GI_MAX_POINT_LIGHTS];
    uint4 clusterDimensions;
    uint4 clusterConfig;
    uint4 renderExtentAndHiZ;
    uint4 cullingConfig;
    float4 cameraNearFar;
    float4 occlusionConfig;
    uint4 occlusionDebugConfig;
    uint4 globalIlluminationConfig;
};

#if defined(IC_TARGET_VULKAN)
[[vk::binding(50, 0)]]
#endif
ConstantBuffer<GiFrameConstants> gGiFrame : register(b1, space0);

// General 4x4 inverse (cofactor expansion) — used to reconstruct world space
// from post-projection depth. Reduced-resolution passes only, so the cost is
// amortized well below the ray-tracing work.
float4x4 giInverse(float4x4 m)
{
    const float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3];
    const float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3];
    const float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3];
    const float a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3];
    const float b00 = a00 * a11 - a01 * a10;
    const float b01 = a00 * a12 - a02 * a10;
    const float b02 = a00 * a13 - a03 * a10;
    const float b03 = a01 * a12 - a02 * a11;
    const float b04 = a01 * a13 - a03 * a11;
    const float b05 = a02 * a13 - a03 * a12;
    const float b06 = a20 * a31 - a21 * a30;
    const float b07 = a20 * a32 - a22 * a30;
    const float b08 = a20 * a33 - a23 * a30;
    const float b09 = a21 * a32 - a22 * a31;
    const float b10 = a21 * a33 - a23 * a31;
    const float b11 = a22 * a33 - a23 * a32;
    float det = b00 * b11 - b01 * b10 + b02 * b09 +
        b03 * b08 - b04 * b07 + b05 * b06;
    if (abs(det) < 1e-20f) return (float4x4)0;
    const float inv = 1.0f / det;
    float4x4 r;
    r[0][0] = ( a11 * b11 - a12 * b10 + a13 * b09) * inv;
    r[0][1] = (-a01 * b11 + a02 * b10 - a03 * b09) * inv;
    r[0][2] = ( a31 * b05 - a32 * b04 + a33 * b03) * inv;
    r[0][3] = (-a21 * b05 + a22 * b04 - a23 * b03) * inv;
    r[1][0] = (-a10 * b11 + a12 * b08 - a13 * b07) * inv;
    r[1][1] = ( a00 * b11 - a02 * b08 + a03 * b07) * inv;
    r[1][2] = (-a30 * b05 + a32 * b02 - a33 * b01) * inv;
    r[1][3] = ( a20 * b05 - a22 * b02 + a23 * b01) * inv;
    r[2][0] = ( a10 * b10 - a11 * b08 + a13 * b06) * inv;
    r[2][1] = (-a00 * b10 + a01 * b08 - a03 * b06) * inv;
    r[2][2] = ( a30 * b04 - a31 * b02 + a33 * b00) * inv;
    r[2][3] = (-a20 * b04 + a21 * b02 - a23 * b00) * inv;
    r[3][0] = (-a10 * b09 + a11 * b07 - a12 * b06) * inv;
    r[3][1] = ( a00 * b09 - a01 * b07 + a02 * b06) * inv;
    r[3][2] = (-a30 * b03 + a31 * b01 - a32 * b00) * inv;
    r[3][3] = ( a20 * b03 - a21 * b01 + a22 * b00) * inv;
    return r;
}

// Reconstruct world position from a hardware depth sample. `uv` is in [0,1];
// `deviceDepth` is the raw depth-buffer value. Handles reversed-Z via the
// frame culling flag already carried in the shared constants.
float3 giWorldFromDepth(float2 uv, float deviceDepth)
{
    float2 ndc = uv * 2.0f - 1.0f;
#if !defined(IC_TARGET_VULKAN)
    ndc.y = -ndc.y;
#endif
    const float4 clip = float4(ndc, deviceDepth, 1.0f);
    const float4x4 invViewProj = giInverse(gGiFrame.viewProjection);
    const float4 world = mul(invViewProj, clip);
    return world.xyz / (abs(world.w) < 1e-8f ? 1e-8f : world.w);
}

#if defined(IC_TARGET_VULKAN)
[[vk::binding(41, 0)]]
#endif
ByteAddressBuffer gGiRtGeometries : register(t41, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(42, 0)]]
#endif
ByteAddressBuffer gGiRtInstances : register(t42, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(43, 0)]]
#endif
ByteAddressBuffer gGiRtMaterials : register(t43, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(44, 0)]]
#endif
ByteAddressBuffer gGiRtVertices[GI_MAX_MODEL_BUFFERS] : register(t44, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(45, 0)]]
#endif
ByteAddressBuffer gGiRtIndices[GI_MAX_MODEL_BUFFERS] : register(t300, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(46, 0)]]
#endif
Texture2D<float4> gGiRtTextures[GI_MAX_TEXTURES] : register(t556, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(47, 0)]]
#endif
SamplerState gGiRtSamplers[GI_MAX_SAMPLERS] : register(s0, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(48, 0)]]
#endif
TextureCube<float4> gGiEnvironment : register(t4652, space0);
#if defined(IC_TARGET_VULKAN)
[[vk::binding(49, 0)]]
#endif
SamplerState gGiEnvironmentSampler : register(s256, space0);

struct GiGeometry
{
    uint4 range;
    uint4 mapping;
    uint4 source;
};

struct GiInstance
{
    float4 objectToWorld[4];
    float4 normalToWorld[4];
    uint4 identity;
    uint4 state;
};

struct GiVertex
{
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uv;
};

struct GiMaterial
{
    float4 baseColor;
    float4 emissive;
    float metallicFactor;
    float roughnessFactor;
    float alphaCutoff;
    float occlusionStrength;
    uint flags;
    uint baseColorTexture;
    uint normalTexture;
    uint metallicRoughnessTexture;
    uint occlusionTexture;
    uint emissiveTexture;
    uint baseColorSampler;
    uint normalSampler;
    uint metallicRoughnessSampler;
    uint occlusionSampler;
    uint emissiveSampler;
};

GiGeometry giLoadGeometry(uint index)
{
    GiGeometry result;
    const uint address = index * GI_GEOMETRY_STRIDE;
    result.range = gGiRtGeometries.Load4(address + 0u);
    result.mapping = gGiRtGeometries.Load4(address + 16u);
    result.source = gGiRtGeometries.Load4(address + 32u);
    return result;
}

GiInstance giLoadInstance(uint index)
{
    GiInstance result;
    const uint address = index * GI_INSTANCE_STRIDE;
    [unroll] for (uint i = 0u; i < 4u; ++i)
        result.objectToWorld[i] = asfloat(gGiRtInstances.Load4(address + i * 16u));
    [unroll] for (uint i = 0u; i < 4u; ++i)
        result.normalToWorld[i] = asfloat(gGiRtInstances.Load4(address + 64u + i * 16u));
    result.identity = gGiRtInstances.Load4(address + 128u);
    result.state = gGiRtInstances.Load4(address + 144u);
    return result;
}

float3 giTransformPoint(GiInstance instance, float3 localPosition)
{
    return instance.objectToWorld[0].xyz * localPosition.x +
        instance.objectToWorld[1].xyz * localPosition.y +
        instance.objectToWorld[2].xyz * localPosition.z +
        instance.objectToWorld[3].xyz;
}

float3 giTransformNormal(GiInstance instance, float3 normal)
{
    return normalize(instance.normalToWorld[0].xyz * normal.x +
        instance.normalToWorld[1].xyz * normal.y +
        instance.normalToWorld[2].xyz * normal.z);
}

float3 giTransformDirection(GiInstance instance, float3 direction)
{
    return normalize(instance.objectToWorld[0].xyz * direction.x +
        instance.objectToWorld[1].xyz * direction.y +
        instance.objectToWorld[2].xyz * direction.z);
}

GiVertex giLoadVertex(uint modelBuffer, uint index)
{
    GiVertex result;
    const uint address = index * GI_VERTEX_STRIDE;
    const uint descriptor = NonUniformResourceIndex(modelBuffer);
    result.position = asfloat(gGiRtVertices[descriptor].Load3(address + 0u));
    result.normal = asfloat(gGiRtVertices[descriptor].Load3(address + 12u));
    result.tangent = asfloat(gGiRtVertices[descriptor].Load4(address + 24u));
    result.uv = asfloat(gGiRtVertices[descriptor].Load2(address + 40u));
    return result;
}

uint3 giLoadTriangleIndices(GiGeometry geometry, uint primitiveIndex)
{
    const uint descriptor = NonUniformResourceIndex(geometry.mapping.x);
    return gGiRtIndices[descriptor].Load3(
        (geometry.range.z + primitiveIndex * 3u) * 4u);
}

GiMaterial giLoadMaterial(uint index)
{
    GiMaterial result;
    const uint address = index * GI_MATERIAL_STRIDE;
    result.baseColor = asfloat(gGiRtMaterials.Load4(address + 0u));
    result.emissive = asfloat(gGiRtMaterials.Load4(address + 16u));
    result.metallicFactor = asfloat(gGiRtMaterials.Load(address + 32u));
    result.roughnessFactor = asfloat(gGiRtMaterials.Load(address + 36u));
    result.alphaCutoff = asfloat(gGiRtMaterials.Load(address + 40u));
    result.occlusionStrength = asfloat(gGiRtMaterials.Load(address + 44u));
    result.flags = gGiRtMaterials.Load(address + 48u);
    result.baseColorTexture = gGiRtMaterials.Load(address + 52u);
    result.normalTexture = gGiRtMaterials.Load(address + 56u);
    result.metallicRoughnessTexture = gGiRtMaterials.Load(address + 60u);
    result.occlusionTexture = gGiRtMaterials.Load(address + 64u);
    result.emissiveTexture = gGiRtMaterials.Load(address + 68u);
    result.baseColorSampler = gGiRtMaterials.Load(address + 72u);
    result.normalSampler = gGiRtMaterials.Load(address + 76u);
    result.metallicRoughnessSampler = gGiRtMaterials.Load(address + 80u);
    result.occlusionSampler = gGiRtMaterials.Load(address + 84u);
    result.emissiveSampler = gGiRtMaterials.Load(address + 88u);
    return result;
}

float2 giBarycentricUv(GiVertex v0, GiVertex v1, GiVertex v2, float2 bary)
{
    const float w = 1.0f - bary.x - bary.y;
    return v0.uv * w + v1.uv * bary.x + v2.uv * bary.y;
}

bool giValidTextureSampler(uint textureIndex, uint samplerIndex)
{
    return textureIndex != GI_INVALID_INDEX && samplerIndex != GI_INVALID_INDEX &&
        textureIndex < min(gGiTraceConstants.textureCount, GI_MAX_TEXTURES) &&
        samplerIndex < min(gGiTraceConstants.samplerCount, GI_MAX_SAMPLERS);
}

float4 giSampleTexture(uint textureIndex, uint samplerIndex, float2 uv)
{
    return gGiRtTextures[NonUniformResourceIndex(textureIndex)].SampleLevel(
        gGiRtSamplers[NonUniformResourceIndex(samplerIndex)], uv, 0.0f);
}

uint giHash(uint value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    return value ^ (value >> 16u);
}

float giRadicalInverse(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

float3 giHemisphereDirection(float3 normal, uint surfelIndex, uint sampleIndex)
{
    // rayOffset is owned by the update command. The current validation slice
    // keeps it stable so identical scene work produces identical Vulkan/DXR
    // rays; the cache scheduler can advance it later without changing tracing.
    const uint sequence = sampleIndex;
    const float2 xi = float2(
        frac((float(sequence) + 0.5f) / 4096.0f +
            float(giHash(surfelIndex)) * 2.3283064365386963e-10f),
        giRadicalInverse(sequence ^ giHash(surfelIndex + 0x9e3779b9u)));
    const float radius = sqrt(xi.x);
    const float angle = 6.28318530718f * xi.y;
    const float3 local = float3(
        radius * cos(angle), radius * sin(angle), sqrt(max(0.0f, 1.0f - xi.x)));
    const float3 up = abs(normal.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f)
                                             : float3(1.0f, 0.0f, 0.0f);
    const float3 tangent = normalize(cross(up, normal));
    const float3 bitangent = cross(normal, tangent);
    return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}

float giOriginOffset(float3 position, float3 normal, float reconstructionError)
{
    const float magnitude = max(abs(position.x), max(abs(position.y), abs(position.z)));
    return max(gGiTraceConstants.rayMinDistance,
        reconstructionError * 2.0f + magnitude * 1.0e-6f +
        2.0e-5f * (1.0f + abs(normal.x) + abs(normal.y) + abs(normal.z)));
}

#endif
