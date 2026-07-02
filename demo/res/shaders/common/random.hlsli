#ifndef IC_RANDOM_HLSLI
#define IC_RANDOM_HLSLI

uint pcgHash(uint value)
{
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random01(inout uint state)
{
    state = pcgHash(state);
    return (float)state * (1.0f / 4294967296.0f);
}

float2 random02(inout uint state)
{
    return float2(random01(state), random01(state));
}

#endif
