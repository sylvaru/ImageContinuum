#include "../common/ic_shader_common.hlsli"
#include "../common/ic_camera.hlsli"
#include "../common/ic_materials.hlsli"
#include "../common/ic_bindless.hlsli"
#include "../common/brdf.hlsli"
#include "../common/clustered_forward.hlsli"

#if defined(IC_TARGET_VULKAN)
[[vk::binding(4098, 0)]]
#endif
TextureCube<float4> gIrradianceTexture : register(t0, space1);

#if defined(IC_TARGET_VULKAN)
[[vk::binding(4099, 0)]]
#endif
TextureCube<float4> gPrefilteredEnvironmentTexture : register(t1, space1);

#if defined(IC_TARGET_VULKAN)
[[vk::binding(4100, 0)]]
#endif
Texture2D<float4> gBrdfLutTexture : register(t2, space1);

#if defined(IC_TARGET_VULKAN)
[[vk::binding(256, 0)]]
#endif
SamplerState gIblSampler : register(s0, space1);

struct VertexInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float4 tangent : TANGENT;
    float2 uv0 : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
    float4 color : COLOR0;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 worldPosition : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv0 : TEXCOORD2;
    float4 color : TEXCOORD3;
    nointerpolation uint materialIndex : TEXCOORD4;
    float4 tangent : TEXCOORD5;
};

VertexOutput VSMain(VertexInput input)
{
    const ObjectData objectData = gObjects[gDraw.objectIndex];

    const float4 worldPosition =
        mul(objectData.world, float4(input.position, 1.0f));

    const float3 worldNormal =
        mul((float3x3) objectData.inverseTransposeWorld, input.normal);

    const float3 worldTangent =
        mul((float3x3) objectData.world, input.tangent.xyz);

    VertexOutput output;
    output.position = mul(gFrame.viewProjection, worldPosition);
    output.worldPosition = worldPosition.xyz;
    output.normal = safeNormalize(worldNormal, float3(0.0f, 1.0f, 0.0f));
    output.uv0 = input.uv0;
    output.color = input.color;
    output.materialIndex = gDraw.materialIndex;
    output.tangent = float4(
        safeNormalize(worldTangent, float3(1.0f, 0.0f, 0.0f)),
        input.tangent.w);

    return output;
}

float4 sampleMaterialTexture(
    uint textureIndex,
    uint samplerIndex,
    float2 uv,
    float4 fallbackValue)
{
#if !defined(IC_DISABLE_TEXTURE_SAMPLING)
    if (textureIndex != IC_INVALID_BINDLESS_INDEX &&
        samplerIndex != IC_INVALID_BINDLESS_INDEX)
    {
        return gBindlessTextures[NonUniformResourceIndex(textureIndex)].Sample(
            gBindlessSamplers[NonUniformResourceIndex(samplerIndex)],
            uv);
    }
#else
    fallbackValue.rgb += 0.0f * float3(textureIndex, samplerIndex, 0.0f);
#endif

    return fallbackValue;
}

float3 evaluateIblAmbient(
    float3 baseColor,
    float metallic,
    float roughness,
    float ao,
    float3 normal,
    float3 viewDirection)
{
    float3 ambient = 0.03f * baseColor * ao;

    if (gFrame.environmentEnabled != 0u)
    {
        const float3 reflection =
            reflect(-viewDirection, normal);

        const float nDotV =
            saturate(dot(normal, viewDirection));

        const float3 f0 =
            lerp(float3(0.04f, 0.04f, 0.04f), baseColor, metallic);

        const float3 fresnel =
            fresnelSchlickRoughness(nDotV, f0, roughness);

        const float3 diffuseIrradiance =
            gIrradianceTexture.SampleLevel(
                gIblSampler,
                normal,
                0.0f).rgb;

        const float maxMip =
            max((float) gFrame.prefilteredMipCount - 1.0f, 0.0f);

        const float3 prefilteredColor =
            gPrefilteredEnvironmentTexture.SampleLevel(
                gIblSampler,
                reflection,
                roughness * maxMip).rgb;

        const float2 brdf =
            gBrdfLutTexture.SampleLevel(
                gIblSampler,
                float2(nDotV, roughness),
                0.0f).rg;

        const float3 diffuse =
            diffuseIrradiance * baseColor * (1.0f - metallic);

        const float3 specular =
            prefilteredColor * (fresnel * brdf.x + brdf.y);

        ambient =
            (diffuse + specular) *
            ao *
            gFrame.environmentIntensity *
            gFrame.environmentExposure;
    }

    return ambient;
}

float4 PSMain(VertexOutput input) : SV_Target0
{
    const MaterialData material = gMaterials[input.materialIndex];

    const float4 sampledBaseColor =
        sampleMaterialTexture(
            material.baseColorTextureIndex,
            material.baseColorSamplerIndex,
            input.uv0,
            float4(1.0f, 1.0f, 1.0f, 1.0f));

    const float3 baseColor =
        sampledBaseColor.rgb *
        material.baseColorFactor.rgb *
        input.color.rgb;

    const float4 normalSample =
        sampleMaterialTexture(
            material.normalTextureIndex,
            material.normalSamplerIndex,
            input.uv0,
            float4(0.5f, 0.5f, 1.0f, 1.0f));

    const float3 geometricNormal =
        safeNormalize(input.normal, float3(0.0f, 1.0f, 0.0f));

    const float3 tangent =
        safeNormalize(
            input.tangent.xyz -
                geometricNormal * dot(input.tangent.xyz, geometricNormal),
            float3(1.0f, 0.0f, 0.0f));

    const float3 bitangent =
        safeNormalize(
            cross(geometricNormal, tangent) * input.tangent.w,
            float3(0.0f, 0.0f, 1.0f));

    const float3 normalMap =
        normalSample.xyz * 2.0f - 1.0f;

    const float3 normal =
        safeNormalize(
            tangent * normalMap.x +
                bitangent * normalMap.y +
                geometricNormal * normalMap.z,
            geometricNormal);

    const float4 metallicRoughnessSample =
        sampleMaterialTexture(
            material.metallicRoughnessTextureIndex,
            material.metallicRoughnessSamplerIndex,
            input.uv0,
            float4(1.0f, 1.0f, 0.0f, 1.0f));

    const float metallic =
        saturate(material.metallicFactor * metallicRoughnessSample.b);

    const float roughness =
        clamp(
            material.roughnessFactor * metallicRoughnessSample.g,
            0.04f,
            1.0f);

    const float ao =
        lerp(
            1.0f,
            sampleMaterialTexture(
                material.occlusionTextureIndex,
                material.occlusionSamplerIndex,
                input.uv0,
                float4(1.0f, 1.0f, 1.0f, 1.0f)).r,
            saturate(material.occlusionStrength));

    const float3 emissive =
        material.emissiveFactor.rgb *
        sampleMaterialTexture(
            material.emissiveTextureIndex,
            material.emissiveSamplerIndex,
            input.uv0,
            float4(1.0f, 1.0f, 1.0f, 1.0f)).rgb;

    const float3 viewDirection =
        safeNormalize(gFrame.cameraPosition - input.worldPosition, normal);

    const float3 lightDirection =
        safeNormalize(-gFrame.lightDirection, float3(0.0f, 1.0f, 0.0f));

    float3 direct =
        pbrDirectLighting(
            baseColor,
            metallic,
            roughness,
            normal,
            viewDirection,
            lightDirection,
            gFrame.lightColor * gFrame.lightIntensity);

    const uint clusterIndex =
        clusteredClusterIndexFromPixel(input.position, input.worldPosition);

    const ClusterLightGrid grid =
        gClusterLightGrid[clusterIndex];

    const uint lightCount =
        min(grid.count, gFrame.clusterConfig.z);

    if (gFrame.clusterConfig.w != 0u)
    {
        uint affectingLightCount = 0u;

        for (uint i = 0; i < lightCount; ++i)
        {
            const uint lightIndex =
                gClusterLightIndices[grid.offset + i];

            if (lightIndex >= gFrame.pointLightCount)
            {
                continue;
            }

            const VisibleLight light =
                gVisibleLights[lightIndex];

            const float3 toLight =
                light.positionRange.xyz - input.worldPosition;

            const float range =
                max(light.positionRange.w, 0.001f);

            if (dot(toLight, toLight) <= range * range)
            {
                affectingLightCount++;
            }
        }

        if (affectingLightCount == 0u)
        {
            return float4(0.0f, 0.0f, 0.0f, 1.0f);
        }

        const float t =
            saturate((float) affectingLightCount / 4.0f);

        const float3 cool =
            float3(0.02f, 0.08f, 0.35f);

        const float3 mid =
            float3(0.05f, 0.85f, 0.25f);

        const float3 hot =
            float3(1.0f, 0.08f, 0.02f);

        const float3 color =
            t < 0.5f
                ? lerp(cool, mid, t * 2.0f)
                : lerp(mid, hot, (t - 0.5f) * 2.0f);

        return float4(color, 1.0f);
    }

    for (uint i = 0; i < lightCount; ++i)
    {
        const uint lightIndex =
            gClusterLightIndices[grid.offset + i];

        if (lightIndex >= gFrame.pointLightCount)
        {
            continue;
        }

        const VisibleLight light =
            gVisibleLights[lightIndex];

        const float3 toLight =
            light.positionRange.xyz - input.worldPosition;

        const float distanceSq =
            max(dot(toLight, toLight), 1.0e-4f);

        const float distance =
            sqrt(distanceSq);

        const float range =
            max(light.positionRange.w, 0.001f);

        const float rangeAttenuation =
            saturate(1.0f - (distance / range) * (distance / range));

        const float attenuation =
            (rangeAttenuation * rangeAttenuation) / distanceSq;

        direct +=
            pbrDirectLighting(
                baseColor,
                metallic,
                roughness,
                normal,
                viewDirection,
                toLight / distance,
                light.colorIntensity.rgb *
                    light.colorIntensity.w *
                    attenuation);
    }

    const float3 ambient =
        evaluateIblAmbient(
            baseColor,
            metallic,
            roughness,
            ao,
            normal,
            viewDirection);

    float3 color =
        ambient + direct + emissive;

    color =
        color / (color + 1.0f);

    color =
        pow(max(color, 0.0f), 1.0f / 2.2f);

    return float4(
        color,
        sampledBaseColor.a * material.baseColorFactor.a);
}