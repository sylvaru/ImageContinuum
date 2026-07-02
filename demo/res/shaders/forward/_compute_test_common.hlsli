RWStructuredBuffer<uint> gComputeTestBuffer : register(u0, space0);

[numthreads(64, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;

#if defined(IC_VISIBILITY_TEST_WRITE)
    gComputeTestBuffer[index] = 0x1c000000u | index;
#else
    const uint value = gComputeTestBuffer[index];
    gComputeTestBuffer[index] = value ^ 0x00ffffffu;
#endif
}
