#include "../common/ray_common.hlsli"
#include "../common/random.hlsli"
#include "../common/scene_bindings.hlsli"
#include "../common/bvh.hlsli"

cbuffer gPathTraceConstants : register(b0, space0)
{
    PathTraceConstants gConstants;
};

RWTexture2D<float4> gAccumulation : register(u1, space0);
StructuredBuffer<PathTraceVertex> gSceneVertices : register(t2, space0);
StructuredBuffer<PathTraceMaterial> gSceneMaterials : register(t3, space0);
StructuredBuffer<PathTraceTriangle> gSceneTriangles : register(t4, space0);
StructuredBuffer<PathTraceBVHNode> gSceneBvhNodes : register(t5, space0);
TextureCube<float4> gEnvironmentTexture : register(t6, space0);
SamplerState gEnvironmentSampler : register(s7, space0);
Texture2D<float4> gBindlessTextures[] : register(t4098, space0);
SamplerState gBindlessSamplers[] : register(s256, space0);

static const uint IC_INVALID_BINDLESS_INDEX = 0xffffffffu;
static const uint MaterialDiffuse = 0;
static const uint MaterialEmissive = 1;

struct Hit
{
    float t;
    float3 position;
    float3 normal;
    float3 albedo;
    float3 emission;
    float metallic;
    float roughness;
    float ao;
    uint materialType;
};

float4 sampleMaterialTexture(uint textureIndex, uint samplerIndex, float2 uv, float4 fallbackValue)
{
    if (textureIndex != IC_INVALID_BINDLESS_INDEX &&
        samplerIndex != IC_INVALID_BINDLESS_INDEX)
    {
        return gBindlessTextures[NonUniformResourceIndex(textureIndex)].SampleLevel(
            gBindlessSamplers[NonUniformResourceIndex(samplerIndex)],
            uv,
            0.0f);
    }

    return fallbackValue;
}

bool intersectPlane(
    Ray ray,
    float3 planePoint,
    float3 normal,
    float2 minA,
    float2 maxA,
    uint axisA,
    uint axisB,
    float3 albedo,
    float3 emission,
    uint materialType,
    inout Hit bestHit)
{
    const float denom = dot(ray.direction, normal);
    if (abs(denom) < 1.0e-5f)
    {
        return false;
    }

    const float t = dot(planePoint - ray.origin, normal) / denom;
    if (t <= 0.001f || t >= bestHit.t)
    {
        return false;
    }

    const float3 position = ray.origin + ray.direction * t;
    const float values[3] = { position.x, position.y, position.z };
    const float2 uv = float2(values[axisA], values[axisB]);
    if (any(uv < minA) || any(uv > maxA))
    {
        return false;
    }

    bestHit.t = t;
    bestHit.position = position;
    bestHit.normal = denom < 0.0f ? normal : -normal;
    bestHit.albedo = albedo;
    bestHit.emission = emission;
    bestHit.metallic = 0.0f;
    bestHit.roughness = 1.0f;
    bestHit.ao = 1.0f;
    bestHit.materialType = materialType;
    return true;
}

bool intersectBox(
    Ray ray,
    float3 boundsMin,
    float3 boundsMax,
    float3 albedo,
    inout Hit bestHit)
{
    const float3 invDirection = 1.0f / ray.direction;
    const float3 t0 = (boundsMin - ray.origin) * invDirection;
    const float3 t1 = (boundsMax - ray.origin) * invDirection;
    const float3 tNear3 = min(t0, t1);
    const float3 tFar3 = max(t0, t1);
    const float tNear = max(max(tNear3.x, tNear3.y), tNear3.z);
    const float tFar = min(min(tFar3.x, tFar3.y), tFar3.z);

    if (tNear > tFar || tFar <= 0.001f || tNear >= bestHit.t)
    {
        return false;
    }

    const float t = tNear > 0.001f ? tNear : tFar;
    if (t >= bestHit.t)
    {
        return false;
    }

    const float3 position = ray.origin + ray.direction * t;
    float3 normal = 0.0f;
    const float3 center = (boundsMin + boundsMax) * 0.5f;
    const float3 extents = (boundsMax - boundsMin) * 0.5f;
    const float3 local = (position - center) / extents;
    const float3 absLocal = abs(local);

    if (absLocal.x > absLocal.y && absLocal.x > absLocal.z)
    {
        normal = float3(sign(local.x), 0.0f, 0.0f);
    }
    else if (absLocal.y > absLocal.z)
    {
        normal = float3(0.0f, sign(local.y), 0.0f);
    }
    else
    {
        normal = float3(0.0f, 0.0f, sign(local.z));
    }

    if (dot(normal, ray.direction) > 0.0f)
    {
        normal = -normal;
    }

    bestHit.t = t;
    bestHit.position = position;
    bestHit.normal = normal;
    bestHit.albedo = albedo;
    bestHit.emission = 0.0f;
    bestHit.metallic = 0.0f;
    bestHit.roughness = 1.0f;
    bestHit.ao = 1.0f;
    bestHit.materialType = MaterialDiffuse;
    return true;
}

Hit makeMiss()
{
    Hit hit;
    hit.t = 1.0e20f;
    hit.position = 0.0f;
    hit.normal = 0.0f;
    hit.albedo = 0.0f;
    hit.emission = 0.0f;
    hit.metallic = 0.0f;
    hit.roughness = 1.0f;
    hit.ao = 1.0f;
    hit.materialType = MaterialDiffuse;
    return hit;
}

bool intersectBounds(
    Ray ray,
    float3 boundsMin,
    float3 boundsMax,
    float maxT)
{
    const float3 invDirection = 1.0f / ray.direction;
    const float3 t0 = (boundsMin - ray.origin) * invDirection;
    const float3 t1 = (boundsMax - ray.origin) * invDirection;
    const float3 tNear3 = min(t0, t1);
    const float3 tFar3 = max(t0, t1);
    const float tNear = max(max(tNear3.x, tNear3.y), tNear3.z);
    const float tFar = min(min(tFar3.x, tFar3.y), tFar3.z);
    return tNear <= tFar && tFar > 0.001f && tNear < maxT;
}

bool intersectSceneTriangle(
    Ray ray,
    uint triangleIndex,
    inout Hit bestHit)
{
    if (triangleIndex >= gConstants.sceneTriangleCount)
    {
        return false;
    }

    const PathTraceTriangle tri = gSceneTriangles[triangleIndex];
    if (tri.i0 >= gConstants.sceneVertexCount ||
        tri.i1 >= gConstants.sceneVertexCount ||
        tri.i2 >= gConstants.sceneVertexCount)
    {
        return false;
    }

    const PathTraceVertex v0 = gSceneVertices[tri.i0];
    const PathTraceVertex v1 = gSceneVertices[tri.i1];
    const PathTraceVertex v2 = gSceneVertices[tri.i2];

    const float3 p0 = v0.position.xyz;
    const float3 p1 = v1.position.xyz;
    const float3 p2 = v2.position.xyz;

    const float3 e1 = p1 - p0;
    const float3 e2 = p2 - p0;
    const float3 p = cross(ray.direction, e2);
    const float det = dot(e1, p);
    if (abs(det) < 1.0e-7f)
    {
        return false;
    }

    const float invDet = 1.0f / det;
    const float3 s = ray.origin - p0;
    const float u = dot(s, p) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }

    const float3 q = cross(s, e1);
    const float v = dot(ray.direction, q) * invDet;
    if (v < 0.0f || u + v > 1.0f)
    {
        return false;
    }

    const float t = dot(e2, q) * invDet;
    if (t <= 0.001f || t >= bestHit.t)
    {
        return false;
    }

    float3 normal =
        safeNormalize(
            (v0.normal * (1.0f - u - v) +
                v1.normal * u +
                v2.normal * v).xyz);
    if (dot(normal, normal) < 1.0e-6f)
    {
        normal = safeNormalize(cross(e1, e2));
    }
    PathTraceMaterial material;
    material.baseColor = float4(0.8f, 0.8f, 0.8f, 1.0f);
    material.emissive = 0.0f;
    material.materialType = MaterialDiffuse;
    material.roughnessFactor = 1.0f;
    material.metallicFactor = 0.0f;
    material.occlusionStrength = 1.0f;
    if (tri.materialIndex < gConstants.sceneMaterialCount)
    {
        material = gSceneMaterials[tri.materialIndex];
    }

    const float w = 1.0f - u - v;
    const float2 uv0 =
        v0.texCoord.xy * w +
        v1.texCoord.xy * u +
        v2.texCoord.xy * v;
    const float3 tangentSample =
        v0.tangent.xyz * w +
        v1.tangent.xyz * u +
        v2.tangent.xyz * v;
    const float tangentSign =
        v0.tangent.w * w +
        v1.tangent.w * u +
        v2.tangent.w * v;

    const float4 baseColorSample =
        sampleMaterialTexture(
            material.baseColorTextureIndex,
            material.baseColorSamplerIndex,
            uv0,
            float4(1.0f, 1.0f, 1.0f, 1.0f));
    const float4 normalSample =
        sampleMaterialTexture(
            material.normalTextureIndex,
            material.normalSamplerIndex,
            uv0,
            float4(0.5f, 0.5f, 1.0f, 1.0f));
    const float4 metallicRoughnessSample =
        sampleMaterialTexture(
            material.metallicRoughnessTextureIndex,
            material.metallicRoughnessSamplerIndex,
            uv0,
            float4(1.0f, 1.0f, 0.0f, 1.0f));
    const float4 occlusionSample =
        sampleMaterialTexture(
            material.occlusionTextureIndex,
            material.occlusionSamplerIndex,
            uv0,
            float4(1.0f, 1.0f, 1.0f, 1.0f));
    const float4 emissiveSample =
        sampleMaterialTexture(
            material.emissiveTextureIndex,
            material.emissiveSamplerIndex,
            uv0,
            float4(1.0f, 1.0f, 1.0f, 1.0f));

    const float3 tangent =
        safeNormalize(tangentSample - normal * dot(tangentSample, normal));
    const float3 bitangent =
        safeNormalize(cross(normal, tangent) * (tangentSign < 0.0f ? -1.0f : 1.0f));
    const float3 normalMap = normalSample.xyz * 2.0f - 1.0f;
    normal =
        safeNormalize(
            tangent * normalMap.x +
            bitangent * normalMap.y +
            normal * normalMap.z);
    if (dot(normal, ray.direction) > 0.0f)
    {
        normal = -normal;
    }

    bestHit.t = t;
    bestHit.position = ray.origin + ray.direction * t;
    bestHit.normal = normal;
    bestHit.albedo =
        max(material.baseColor.rgb * baseColorSample.rgb, 0.0f);
    bestHit.emission =
        max(material.emissive.rgb * emissiveSample.rgb, 0.0f);
    bestHit.metallic =
        saturate(material.metallicFactor * metallicRoughnessSample.b);
    bestHit.roughness =
        clamp(material.roughnessFactor * metallicRoughnessSample.g, 0.04f, 1.0f);
    bestHit.ao =
        lerp(1.0f, occlusionSample.r, saturate(material.occlusionStrength));
    bestHit.materialType =
        dot(bestHit.emission, bestHit.emission) > 0.0f
            ? MaterialEmissive
            : MaterialDiffuse;
    return true;
}

Hit traceBvhScene(Ray ray)
{
    Hit hit = makeMiss();
    if (gConstants.sceneBvhNodeCount == 0u ||
        gConstants.sceneTriangleCount == 0u)
    {
        return hit;
    }

    uint stack[64];
    uint stackSize = 0;
    stack[stackSize++] = 0;

    while (stackSize != 0)
    {
        const uint nodeIndex = stack[--stackSize];
        if (nodeIndex >= gConstants.sceneBvhNodeCount)
        {
            continue;
        }

        const PathTraceBVHNode node = gSceneBvhNodes[nodeIndex];
        if (!intersectBounds(ray, node.boundsMin, node.boundsMax, hit.t))
        {
            continue;
        }

        if (node.count != 0u)
        {
            for (uint i = 0; i < node.count; ++i)
            {
                intersectSceneTriangle(ray, node.leftFirst + i, hit);
            }
            continue;
        }

        const uint left = node.leftFirst;
        const uint right = left + 1u;
        if (stackSize + 2u <= 64u)
        {
            stack[stackSize++] = right;
            stack[stackSize++] = left;
        }
    }

    return hit;
}

Hit traceProceduralScene(Ray ray)
{
    Hit hit = makeMiss();

    const float3 white = float3(0.82f, 0.80f, 0.74f);
    const float3 red = float3(0.70f, 0.10f, 0.08f);
    const float3 green = float3(0.10f, 0.48f, 0.16f);
    const float3 lightEmission = float3(18.0f, 15.0f, 11.0f);

    intersectPlane(ray, float3(-1.0f, 0.0f, 0.0f), float3(1.0f, 0.0f, 0.0f), float2(0.0f, 0.0f), float2(2.0f, 2.0f), 1, 2, red, 0.0f, MaterialDiffuse, hit);
    intersectPlane(ray, float3(1.0f, 0.0f, 0.0f), float3(-1.0f, 0.0f, 0.0f), float2(0.0f, 0.0f), float2(2.0f, 2.0f), 1, 2, green, 0.0f, MaterialDiffuse, hit);
    intersectPlane(ray, float3(0.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float2(-1.0f, 0.0f), float2(1.0f, 2.0f), 0, 2, white, 0.0f, MaterialDiffuse, hit);
    intersectPlane(ray, float3(0.0f, 2.0f, 0.0f), float3(0.0f, -1.0f, 0.0f), float2(-1.0f, 0.0f), float2(1.0f, 2.0f), 0, 2, white, 0.0f, MaterialDiffuse, hit);
    intersectPlane(ray, float3(0.0f, 0.0f, 2.0f), float3(0.0f, 0.0f, -1.0f), float2(-1.0f, 0.0f), float2(1.0f, 2.0f), 0, 1, white, 0.0f, MaterialDiffuse, hit);
    intersectPlane(ray, float3(0.0f, 1.995f, 1.0f), float3(0.0f, -1.0f, 0.0f), float2(-0.32f, 0.72f), float2(0.32f, 1.28f), 0, 2, 1.0f, lightEmission, MaterialEmissive, hit);

    intersectBox(ray, float3(-0.75f, 0.0f, 0.25f), float3(-0.18f, 0.68f, 0.82f), white, hit);
    intersectBox(ray, float3(0.18f, 0.0f, 0.70f), float3(0.72f, 1.08f, 1.38f), white, hit);

    return hit;
}

Hit traceScene(Ray ray)
{
    if (gConstants.useSceneGeometry != 0u)
    {
        return traceBvhScene(ray);
    }

    return traceProceduralScene(ray);
}

float3 cosineHemisphere(float3 normal, inout uint rngState)
{
    const float2 xi = random02(rngState);
    const float r = sqrt(xi.x);
    const float phi = 6.28318530718f * xi.y;
    const float x = r * cos(phi);
    const float z = r * sin(phi);
    const float y = sqrt(max(0.0f, 1.0f - xi.x));

    const float3 tangent =
        abs(normal.y) < 0.999f
            ? safeNormalize(cross(float3(0.0f, 1.0f, 0.0f), normal))
            : float3(1.0f, 0.0f, 0.0f);
    const float3 bitangent = cross(normal, tangent);

    return safeNormalize(tangent * x + normal * y + bitangent * z);
}

bool visibleToLight(float3 origin, float3 lightPoint)
{
    Ray shadowRay;
    shadowRay.origin = origin;
    shadowRay.direction = safeNormalize(lightPoint - origin);

    const float lightDistance = length(lightPoint - origin);
    const Hit shadowHit = traceScene(shadowRay);

    return shadowHit.t > lightDistance - 0.01f ||
        shadowHit.materialType == MaterialEmissive;
}

float3 sampleSceneDirectLight(Hit hit, inout uint rngState)
{
    const uint lightTriangleIndex = gConstants.sceneEmissiveTriangleIndex;
    const uint lightTriangleCount = gConstants.sceneEmissiveTriangleCount;
    if (lightTriangleIndex == IC_INVALID_BINDLESS_INDEX ||
        lightTriangleIndex >= gConstants.sceneTriangleCount ||
        lightTriangleCount == 0u ||
        lightTriangleIndex + lightTriangleCount > gConstants.sceneTriangleCount)
    {
        return 0.0f;
    }

    const uint selectedLight = min(
        (uint)(random01(rngState) * lightTriangleCount),
        lightTriangleCount - 1u);
    const PathTraceTriangle tri =
        gSceneTriangles[lightTriangleIndex + selectedLight];
    if (tri.materialIndex >= gConstants.sceneMaterialCount)
    {
        return 0.0f;
    }

    const PathTraceMaterial lightMaterial =
        gSceneMaterials[tri.materialIndex];
    if (dot(lightMaterial.emissive.rgb, lightMaterial.emissive.rgb) <= 0.0f)
    {
        return 0.0f;
    }

    const PathTraceVertex v0 = gSceneVertices[tri.i0];
    const PathTraceVertex v1 = gSceneVertices[tri.i1];
    const PathTraceVertex v2 = gSceneVertices[tri.i2];

    const float2 xi = random02(rngState);
    const float su = sqrt(xi.x);
    const float b0 = 1.0f - su;
    const float b1 = xi.y * su;
    const float b2 = 1.0f - b0 - b1;

    const float3 p0 = v0.position.xyz;
    const float3 p1 = v1.position.xyz;
    const float3 p2 = v2.position.xyz;
    const float3 lightPoint =
        p0 * b0 + p1 * b1 + p2 * b2;
    const float3 lightNormal =
        safeNormalize(cross(p1 - p0, p2 - p0));
    const float area =
        0.5f * length(cross(p1 - p0, p2 - p0));

    const float3 toLight = lightPoint - hit.position;
    const float distanceSq = max(dot(toLight, toLight), 1.0e-4f);
    const float distance = sqrt(distanceSq);
    const float3 lightDir = toLight / distance;

    const float nDotL = max(dot(hit.normal, lightDir), 0.0f);
    const float lightDot = abs(dot(lightNormal, -lightDir));
    if (nDotL <= 0.0f || lightDot <= 0.0f || area <= 0.0f)
    {
        return 0.0f;
    }

    if (!visibleToLight(hit.position + hit.normal * 0.002f, lightPoint))
    {
        return 0.0f;
    }

    const float3 brdf = hit.albedo * hit.ao * (1.0f - hit.metallic) * 0.31830988618f;
    return brdf * lightMaterial.emissive.rgb * nDotL * lightDot *
        (area * float(lightTriangleCount)) / distanceSq;
}

float3 sampleDirectLight(Hit hit, inout uint rngState)
{
    float3 radiance = 0.0f;

    if (gConstants.useSceneGeometry != 0u)
    {
        radiance += sampleSceneDirectLight(hit, rngState);
    }

    const uint pointLightCount =
        min(gConstants.pointLightCount, IC_MAX_PATH_TRACE_POINT_LIGHTS);
    if (pointLightCount != 0u)
    {
        const uint lightIndex =
            min((uint)(random01(rngState) * pointLightCount), pointLightCount - 1u);
        const float4 positionRange =
            gConstants.pointLightPositionRange[lightIndex];
        const float4 colorIntensity =
            gConstants.pointLightColorIntensity[lightIndex];

        const float3 toLight = positionRange.xyz - hit.position;
        const float distanceSq = max(dot(toLight, toLight), 1.0e-4f);
        const float distance = sqrt(distanceSq);
        const float range = max(positionRange.w, 0.001f);
        const float3 lightDir = toLight / distance;
        const float rangeAttenuation =
            saturate(1.0f - (distance / range) * (distance / range));
        const float attenuation =
            (rangeAttenuation * rangeAttenuation) / distanceSq;
        const float nDotL = max(dot(hit.normal, lightDir), 0.0f);
        if (nDotL > 0.0f &&
            attenuation > 0.0f &&
            visibleToLight(hit.position + hit.normal * 0.002f, positionRange.xyz))
        {
            const float3 brdf =
                hit.albedo * hit.ao * (1.0f - hit.metallic) * 0.31830988618f;
            radiance +=
                brdf *
                colorIntensity.rgb *
                colorIntensity.w *
                attenuation *
                nDotL *
                pointLightCount;
        }
    }

    if (gConstants.useSceneGeometry != 0u)
    {
        return radiance;
    }

    const float2 xi = random02(rngState);
    const float3 lightPoint =
        float3(lerp(-0.32f, 0.32f, xi.x), 1.994f, lerp(0.72f, 1.28f, xi.y));
    const float3 toLight = lightPoint - hit.position;
    const float distanceSq = max(dot(toLight, toLight), 1.0e-4f);
    const float distance = sqrt(distanceSq);
    const float3 lightDir = toLight / distance;

    const float nDotL = max(dot(hit.normal, lightDir), 0.0f);
    const float lightDot = max(dot(float3(0.0f, -1.0f, 0.0f), -lightDir), 0.0f);
    if (nDotL <= 0.0f || lightDot <= 0.0f)
    {
        return 0.0f;
    }

    if (!visibleToLight(hit.position + hit.normal * 0.002f, lightPoint))
    {
        return 0.0f;
    }

    const float area = 0.64f * 0.56f;
    const float3 lightEmission = float3(18.0f, 15.0f, 11.0f);
    const float3 brdf = hit.albedo * hit.ao * (1.0f - hit.metallic) * 0.31830988618f;
    return radiance + brdf * lightEmission * nDotL * lightDot * area / distanceSq;
}

float3 clampLuminance(float3 value, float maxLuminance)
{
    const float luminance =
        dot(max(value, 0.0f), float3(0.2126f, 0.7152f, 0.0722f));
    if (luminance <= maxLuminance)
    {
        return value;
    }

    return value * (maxLuminance / max(luminance, 1.0e-4f));
}

float3 environmentRadiance(float3 direction)
{
    if (gConstants.environmentEnabled != 0u)
    {
        return max(
            gEnvironmentTexture.SampleLevel(
                gEnvironmentSampler,
                direction,
                0.0f).rgb,
            0.0f) *
            (gConstants.environmentIntensity * gConstants.environmentExposure);
    }

    // Authored scenes with explicit geometry may intentionally disable the
    // environment (for example, a light-box reference). The procedural sky is
    // retained only for the shader's geometry-free fallback scene.
    return gConstants.useSceneGeometry != 0u
        ? 0.0f
        : skyColor(direction) * 0.015f;
}

Ray makeCameraRay(uint2 pixel, uint rngSeed)
{
    uint localSeed = rngSeed;
    const float2 jitter = random02(localSeed) - 0.5f;
    const float2 uv =
        (float2(pixel) + 0.5f + jitter) /
        max(float2(gConstants.renderSize), 1.0f);
    const float2 ndc = uv * 2.0f - 1.0f;
    const float aspect =
        gConstants.cameraForwardAndAspect.w > 0.0f
            ? gConstants.cameraForwardAndAspect.w
            : (float)gConstants.renderSize.x /
                max((float)gConstants.renderSize.y, 1.0f);

    const float3 origin = gConstants.cameraPositionAndTanHalfFov.xyz;
    const float3 forward = safeNormalize(gConstants.cameraForwardAndAspect.xyz);
    const float3 right = safeNormalize(gConstants.cameraRightAndNear.xyz);
    const float3 up = safeNormalize(gConstants.cameraUpAndFar.xyz);
    const float tanHalfFov = max(gConstants.cameraPositionAndTanHalfFov.w, 0.001f);

    Ray ray;
    ray.origin = origin;
    ray.direction =
        safeNormalize(
            forward +
            right * ndc.x * aspect * tanHalfFov -
            up * ndc.y * tanHalfFov);
    return ray;
}

float3 tracePath(Ray ray, inout uint rngState)
{
    float3 radiance = 0.0f;
    float3 throughput = 1.0f;
    const uint maxBounces = clamp(gConstants.maxBounces, 1u, 8u);

    for (uint bounce = 0; bounce < maxBounces; ++bounce)
    {
        Hit hit = traceScene(ray);
        if (hit.t >= 1.0e19f)
        {
            // A miss always sees the scene environment. Reference mode only
            // changes which surface-lighting bounce is accumulated; it must
            // not turn a valid environment into a black background.
            radiance += throughput * environmentRadiance(ray.direction);
            break;
        }

        if (hit.materialType == MaterialEmissive)
        {
            if ((gConstants.referenceMode == 0u && bounce == 0u) ||
                (gConstants.referenceMode != 0u && bounce == 1u))
            {
                radiance += throughput * hit.emission;
            }
            break;
        }

        if (gConstants.referenceMode == 0u || bounce == 1u)
            radiance += clampLuminance(
                throughput * sampleDirectLight(hit, rngState),
                bounce == 0u ? 4.0f : 2.0f);
        ray.origin = hit.position + hit.normal * 0.002f;
        if (hit.metallic > 0.5f)
        {
            const float3 reflected = reflect(ray.direction, hit.normal);
            const float3 roughDirection = cosineHemisphere(hit.normal, rngState);
            ray.direction =
                safeNormalize(lerp(reflected, roughDirection, hit.roughness * hit.roughness));
            throughput *=
                lerp(float3(0.04f, 0.04f, 0.04f), hit.albedo, hit.metallic) *
                hit.ao;
        }
        else
        {
            throughput *= hit.albedo * hit.ao;
            ray.direction = cosineHemisphere(hit.normal, rngState);
        }
    }

    return radiance;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gConstants.renderSize))
    {
        return;
    }

    uint rngState =
        pcgHash(pixel.x + pixel.y * 4099u) ^
        pcgHash(gConstants.frameIndex + 17u);

    float3 currentSample = 0.0f;
    const uint samplesPerPixel = max(gConstants.samplesPerPixel, 1u);
    for (uint sampleIndex = 0; sampleIndex < samplesPerPixel; ++sampleIndex)
    {
        const Ray ray = makeCameraRay(pixel, rngState);
        currentSample += tracePath(ray, rngState);
        rngState = pcgHash(rngState + sampleIndex + 1u);
    }
    currentSample /= (float)samplesPerPixel;

    if (gConstants.resetAccumulation != 0u ||
        gConstants.accumulatedSampleCount == 0u)
    {
        gAccumulation[pixel] = float4(currentSample, 1.0f);
        return;
    }

    const float previousSampleCount = (float)gConstants.accumulatedSampleCount;
    const float3 previous = gAccumulation[pixel].rgb;
    const float3 accumulated =
        (previous * previousSampleCount + currentSample) /
        (previousSampleCount + 1.0f);

    gAccumulation[pixel] = float4(accumulated, 1.0f);
}
