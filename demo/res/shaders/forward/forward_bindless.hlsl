#include "../common/ic_shader_common.hlsli"
#include "../common/ic_camera.hlsli"
#include "../common/ic_materials.hlsli"
#include "../common/ic_bindless.hlsli"
#include "../common/brdf.hlsli"

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
};

VertexOutput VSMain(VertexInput input)
{
    const ObjectData objectData = gObjects[gDraw.objectIndex];

    const float4 worldPosition =
        mul(objectData.world, float4(input.position, 1.0f));
    const float3 worldNormal =
        mul((float3x3)objectData.inverseTransposeWorld, input.normal);

    VertexOutput output;
    output.position = mul(gFrame.viewProjection, worldPosition);
    output.worldPosition = worldPosition.xyz;
    output.normal = safeNormalize(worldNormal, float3(0.0f, 1.0f, 0.0f));
    output.uv0 = input.uv0;
    output.color = input.color;
    output.materialIndex = gDraw.materialIndex;
    return output;
}

float4 PSMain(VertexOutput input) : SV_Target0
{
    const MaterialData material = gMaterials[input.materialIndex];
    const uint textureIndex = material.baseColorTextureIndex;
    const uint samplerIndex = material.samplerIndex;

    float4 sampledBaseColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
#if !defined(IC_DISABLE_TEXTURE_SAMPLING)
    sampledBaseColor =
        gBindlessTextures[textureIndex].Sample(gBindlessSamplers[samplerIndex], input.uv0);
#else
    sampledBaseColor.rgb += 0.0f * float3(textureIndex, samplerIndex, 0.0f);
#endif

    const float3 baseColor =
        sampledBaseColor.rgb *
        material.baseColorFactor.rgb *
        input.color.rgb;

    const float3 normal = safeNormalize(input.normal, float3(0.0f, 1.0f, 0.0f));
    const float3 lightDirection = safeNormalize(-gFrame.lightDirection, float3(0.0f, 1.0f, 0.0f));
    const float ndotl = saturate(dot(normal, lightDirection));

    const float3 ambient = 0.035f * baseColor;
    const float3 direct =
        lambertDiffuse(baseColor) *
        ndotl *
        gFrame.lightColor *
        gFrame.lightIntensity;
    float3 color = ambient + direct;
    color = color / (color + 1.0f);
    color = pow(max(color, 0.0f), 1.0f / 2.2f);

    return float4(color, sampledBaseColor.a * material.baseColorFactor.a);
}
