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
};

VertexOutput VSMain(VertexInput input, uint instanceId : SV_InstanceID)
{
    const DrawMetadata draw = resolveDrawMetadata(instanceId);
    const ObjectData objectData = gObjects[draw.transformIndex];

    const float4 worldPosition =
        mul(objectData.world, float4(input.position, 1.0f));

    VertexOutput output;
    output.position = mul(gFrame.viewProjection, worldPosition);
    return output;
}

void PSMain(VertexOutput input)
{
}
