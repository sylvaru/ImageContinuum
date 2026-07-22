#include "_gi_common.hlsli"
[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) { giPreserveProbes(id); }
