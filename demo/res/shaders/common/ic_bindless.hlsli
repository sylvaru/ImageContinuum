#ifndef IC_BINDLESS_HLSLI
#define IC_BINDLESS_HLSLI

Texture2D<float4> gBindlessTextures[] : register(t2, space0);
SamplerState gBindlessSamplers[] : register(s0, space0);

static const uint IC_BINDLESS_WHITE_TEXTURE = 0;
static const uint IC_BINDLESS_BLACK_TEXTURE = 1;
static const uint IC_BINDLESS_FLAT_NORMAL_TEXTURE = 2;
static const uint IC_BINDLESS_METALLIC_ROUGHNESS_TEXTURE = 3;
static const uint IC_BINDLESS_DEFAULT_SAMPLER = 0;

#endif
