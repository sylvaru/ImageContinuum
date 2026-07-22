#include "../common/ic_shader_common.hlsli"
#include "../common/ic_camera.hlsli"
#include "../common/ic_materials.hlsli"

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
    float3 worldNormal : TEXCOORD0;
    nointerpolation uint materialIndex : TEXCOORD1;
    nointerpolation uint instanceIndex : TEXCOORD2;
};

VertexOutput VSMain(VertexInput input, uint instanceId : SV_InstanceID)
{
    const DrawMetadata draw = resolveDrawMetadata(instanceId);
    const ObjectData objectData = gObjects[draw.transformIndex];

    const float4 worldPosition =
        mul(objectData.world, float4(input.position, 1.0f));

    VertexOutput output;
    output.position = mul(gFrame.viewProjection, worldPosition);
    output.worldNormal = safeNormalize(
        mul((float3x3)objectData.inverseTransposeWorld, input.normal),
        float3(0.0f, 1.0f, 0.0f));
    output.materialIndex = draw.materialIndex;
    output.instanceIndex = draw.instanceIndex;
    return output;
}

float2 encodeOctahedralNormal(float3 normal)
{
    float3 n = safeNormalize(normal, float3(0.0f, 1.0f, 0.0f));
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    const float2 signs = float2(n.x >= 0.0f ? 1.0f : -1.0f,
        n.y >= 0.0f ? 1.0f : -1.0f);
    return n.z >= 0.0f ? n.xy : (1.0f - abs(n.yx)) * signs;
}

float4 PSMain(VertexOutput input) : SV_Target0
{
    // RGBA32F represents integer scene IDs exactly through 2^24. Encoding the
    // geometric normal into two channels avoids truncating or aliasing either
    // identity field.
    return float4(encodeOctahedralNormal(input.worldNormal),
        float(input.materialIndex), float(input.instanceIndex));
}
