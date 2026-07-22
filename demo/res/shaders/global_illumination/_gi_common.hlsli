#ifndef IC_GI_PLACEHOLDER_COMMON_HLSLI
#define IC_GI_PLACEHOLDER_COMMON_HLSLI

#include "_gi_ray_query_common.hlsli"

RWByteAddressBuffer gGiSurfaceData : register(u0, space0);
RWByteAddressBuffer gGiSurfels : register(u1, space0);
RWByteAddressBuffer gGiVisibilityMoments : register(u2, space0);
RWByteAddressBuffer gGiHashBuckets : register(u3, space0);
RWByteAddressBuffer gGiProbes : register(u4, space0);
RWByteAddressBuffer gGiAllocationQueue : register(u5, space0);
RWByteAddressBuffer gGiPriorityQueue : register(u6, space0);
RWByteAddressBuffer gGiIndirectArguments : register(u7, space0);
RWByteAddressBuffer gGiCounters : register(u8, space0);
RWByteAddressBuffer gGiDiagnostics : register(u9, space0);
RWByteAddressBuffer gGiResidualInterface : register(u10, space0);
RWByteAddressBuffer gGiHitRecords : register(u11, space0);
RWByteAddressBuffer gGiProbeStaging : register(u12, space0);
Texture2D<float4> gGiPrimaryInput : register(t20, space0);
Texture2D<float4> gGiHistoryInput : register(t21, space0);
Texture2D<float> gGiSceneDepth : register(t22, space0);
Texture2D<float4> gGiSurfaceAttributes : register(t23, space0);
Texture2D<float> gGiPreviousSceneDepth : register(t24, space0);
Texture2D<float4> gGiPreviousSurfaceAttributes : register(t25, space0);
RWTexture2D<float4> gGiOutput : register(u30, space0);
RaytracingAccelerationStructure gGiSceneTlas : register(t40, space0);

static const uint GI_TRACE_BASE = 128u;
static const uint GI_TRACE_STRIDE = 800u;
static const uint GI_TRACE_PROBE_BASE = 160u;
static const uint GI_TRACE_PROBE_STRIDE = 80u;

int giTracePixelSlot(uint2 pixel, uint2 dimensions)
{
    const uint y = (dimensions.y * 2u) / 3u;
    if (pixel.y != y) return -1;
    if (pixel.x == dimensions.x / 4u) return 0;
    if (pixel.x == dimensions.x / 2u) return 1;
    if (pixel.x == (dimensions.x * 3u) / 4u) return 2;
    return -1;
}

uint giTracePixelAddress(int slot)
{
    return GI_TRACE_BASE + uint(slot) * GI_TRACE_STRIDE;
}

void storeIfInRange(RWByteAddressBuffer target, uint offset, uint value)
{
    target.Store(offset, value);
}

void giAtomicSaturatingDecrement(RWByteAddressBuffer target, uint offset)
{
    uint observed = target.Load(offset);
    while (observed != 0u)
    {
        uint original = 0u;
        target.InterlockedCompareExchange(offset, observed,
            observed - 1u, original);
        if (original == observed) return;
        observed = original;
    }
}

float giPackCoverageVisibility(float coverage, float visibility)
{
    const uint coverageQ = (uint)round(saturate(coverage) * 1023.0f);
    const uint visibilityQ = (uint)round(saturate(visibility) * 1023.0f);
    return float(coverageQ) + float(visibilityQ) * (1.0f / 1024.0f);
}

void giUnpackCoverageVisibility(float packed, out float coverage,
    out float visibility)
{
    const float integerPart = floor(max(packed, 0.0f));
    coverage = saturate(integerPart * (1.0f / 1023.0f));
    visibility = saturate(frac(max(packed, 0.0f)) *
        (1024.0f / 1023.0f));
}

// Reduced-resolution GI surface grid derived from the full-res scene depth.
uint2 giReducedDims()
{
    uint fw = 0, fh = 0;
    gGiSceneDepth.GetDimensions(fw, fh);
    const uint divisor = max(gGiTraceConstants.evaluationDivisor, 1u);
    return uint2(max((fw + divisor - 1u) / divisor, 1u),
                 max((fh + divisor - 1u) / divisor, 1u));
}

struct GiSurface
{
    float3 position;
    float3 normal;
    float deviceDepth;
    uint materialId;
    uint instanceId;
    bool valid;
};

float3 giDecodeOctahedralNormal(float2 encoded)
{
    float3 n = float3(encoded, 1.0f - abs(encoded.x) - abs(encoded.y));
    if (n.z < 0.0f)
    {
        const float2 signs = float2(n.x >= 0.0f ? 1.0f : -1.0f,
            n.y >= 0.0f ? 1.0f : -1.0f);
        n.xy = (1.0f - abs(n.yx)) * signs;
    }
    return normalize(n);
}

GiSurface giLoadSurface(uint index)
{
    GiSurface s;
    const uint offset = index * 32u;
    const uint4 a = gGiSurfaceData.Load4(offset + 0u);
    const uint4 b = gGiSurfaceData.Load4(offset + 16u);
    s.position = asfloat(a.xyz);
    s.deviceDepth = asfloat(a.w);
    s.normal = asfloat(b.xyz);
    s.valid = b.w != 0u;
    const uint identity = s.valid ? b.w - 1u : 0u;
    s.materialId = identity & 65535u;
    s.instanceId = identity >> 16u;
    return s;
}

void giSurfacePrepare(uint3 id)
{
    // Single per-frame reset point. Surface preparation runs first and is
    // barrier-separated from every later pass, so zeroing the transient counters
    // and diagnostics here is race-free (no other surface-prepare thread touches
    // these buffers).
    if (id.x == 0u)
    {
        [unroll] for (uint c = 0u; c < 64u; c += 4u)
            storeIfInRange(gGiCounters, c, 0u);
        [unroll] for (uint d = 0u; d < 128u; d += 4u)
            storeIfInRange(gGiDiagnostics, d, 0u);
        gGiDiagnostics.Store(0u, min(gGiResidualInterface.Load(0u),
            gGiTraceConstants.maxSurfels));
        gGiDiagnostics.Store(40u, gGiTraceConstants.frameIndex);
        gGiDiagnostics.Store(112u, gGiResidualInterface.Load(4u));
    }

    const uint offset = id.x * 32u;
    const uint surfaceCount =
        gGiTraceConstants.reducedWidth * gGiTraceConstants.reducedHeight;
    if (id.x >= surfaceCount) return;

    uint fw = 0, fh = 0;
    gGiSceneDepth.GetDimensions(fw, fh);
    const uint divisor = max(gGiTraceConstants.evaluationDivisor, 1u);
    const uint2 dims = giReducedDims();
    const uint px = id.x % dims.x;
    const uint py = id.x / dims.x;
    const bool inRange = py < dims.y;

    int2 c = int2(px * divisor + divisor / 2u, py * divisor + divisor / 2u);
    c = min(c, int2(int(fw) - 1, int(fh) - 1));
    const float d = inRange ? gGiSceneDepth.Load(int3(c, 0)) : 0.0f;
    const bool reversedZ = gGiFrame.renderExtentAndHiZ.w != 0u;
    const bool background = !inRange ||
        (reversedZ ? (d <= 0.0f) : (d >= 1.0f));
    if (background)
    {
        gGiSurfaceData.Store4(offset + 0u, uint4(0u, 0u, 0u, 0u));
        gGiSurfaceData.Store4(offset + 16u, uint4(0u, 0u, 0u, 0u));
        return;
    }

    const float2 uv  = (float2(c) + 0.5f) / float2(fw, fh);
    const float3 p  = giWorldFromDepth(uv, d);
    const float4 attributes = gGiSurfaceAttributes.Load(int3(c, 0));
    const float3 n = giDecodeOctahedralNormal(attributes.xy);
    const uint materialId = (uint)round(max(attributes.z, 0.0f));
    const uint instanceId = (uint)round(max(attributes.w, 0.0f));
    const bool finiteSurface = all(isfinite(p)) && all(isfinite(n)) &&
        dot(n, n) > 0.99f && materialId < 65535u && instanceId < 65535u &&
        materialId < gGiTraceConstants.materialCount &&
        instanceId < gGiTraceConstants.instanceCount;
    if (!finiteSurface)
    {
        gGiSurfaceData.Store4(offset + 0u, uint4(0u, 0u, 0u, 0u));
        gGiSurfaceData.Store4(offset + 16u, uint4(0u, 0u, 0u, 0u));
        return;
    }
    const uint identity = (instanceId << 16u) | materialId;

    gGiSurfaceData.Store4(offset + 0u, uint4(asuint(p), asuint(d)));
    gGiSurfaceData.Store4(offset + 16u, uint4(asuint(n), identity + 1u));
}

void giInitializeCache(uint3 id)
{
    uint hashBytes = 0u;
    gGiHashBuckets.GetDimensions(hashBytes);
    if (id.x * 16u < hashBytes)
        gGiHashBuckets.Store4(id.x * 16u,
            uint4(GI_INVALID_INDEX, GI_INVALID_INDEX,
                GI_INVALID_INDEX, GI_INVALID_INDEX));
    uint probeBytes = 0u;
    gGiProbes.GetDimensions(probeBytes);
    if (id.x * 128u < probeBytes)
    {
        const uint probeAddress = id.x * 128u;
        gGiProbes.Store4(probeAddress + 0u, 0u);
        gGiProbeStaging.Store4(probeAddress + 0u, 0u);
        gGiProbes.Store4(probeAddress + 16u, 0u);
        gGiProbeStaging.Store4(probeAddress + 16u, 0u);
        gGiProbes.Store4(probeAddress + 32u, 0u);
        gGiProbeStaging.Store4(probeAddress + 32u, 0u);
        gGiProbes.Store4(probeAddress + 48u, 0u);
        gGiProbeStaging.Store4(probeAddress + 48u, 0u);
        gGiProbes.Store4(probeAddress + 64u, 0u);
        gGiProbeStaging.Store4(probeAddress + 64u, 0u);
        gGiProbes.Store4(probeAddress + 80u, 0u);
        gGiProbeStaging.Store4(probeAddress + 80u, 0u);
        gGiProbes.Store4(probeAddress + 96u, 0u);
        gGiProbeStaging.Store4(probeAddress + 96u, 0u);
        gGiProbes.Store4(probeAddress + 112u, 0u);
        gGiProbeStaging.Store4(probeAddress + 112u, 0u);
    }
    if (id.x == 0u)
    {
        [unroll] for (uint offset = 0u; offset < 64u; offset += 4u)
            storeIfInRange(gGiResidualInterface, offset, 0u);
        gGiResidualInterface.Store(8u, gGiTraceConstants.frameIndex);
    }
}

// ------------------------------------------------------------------
// Persistent world-space hash-grid surfel cache
// ------------------------------------------------------------------
uint giClipmapCount()
{
    return clamp((gGiTraceConstants.giFlags >> 4u) & 0xfu, 1u, 8u);
}

uint giProbeResolution()
{
    return clamp((gGiTraceConstants.giFlags >> 8u) & 0x7fu, 2u, 64u);
}

bool giProbeFallbackEnabled()
{
    return (gGiTraceConstants.giFlags & (1u << 20u)) != 0u;
}

float giLevelCellSize(uint level)
{
    return gGiTraceConstants.cellSize * exp2(float(level));
}

uint giSurfelClipLevel(float3 position)
{
    // Nested, world-stable grids selected relative to the camera. The first
    // tier spans a useful local room around the viewer; farther tiers double
    // support without embedding scene-specific world dimensions.
    // Probe spacing is four base cells. Keep each level active through most of
    // its half extent; the old fixed 16-cell switch selected 4 m probes at only
    // 4 m in the default map and leaked across doors and neighboring rooms.
    const float nearExtent = max(gGiTraceConstants.cellSize *
        float(giProbeResolution()) * 1.6f, 1.0e-3f);
    const float ratio = max(length(position - gGiFrame.cameraPosition) /
        nearExtent, 1.0f);
    return min((uint)floor(log2(ratio)), giClipmapCount() - 1u);
}

int3 giCellCoordAtLevel(float3 position, uint level)
{
    return int3(floor(position / giLevelCellSize(level)));
}

uint giCellKeyAtLevel(int3 cell, uint level)
{
    // Non-zero hash of the signed cell coordinate. Never returns INVALID.
    const uint x = giHash(uint(cell.x * 73856093));
    const uint y = giHash(uint(cell.y * 19349663));
    const uint z = giHash(uint(cell.z * 83492791));
    const uint l = giHash((level + 1u) * 0x9e3779b9u);
    const uint key = giHash(x ^ y ^ z ^ l);
    return key == GI_INVALID_INDEX ? 0u : key;
}

float giSceneRayMinDistance()
{
    return max(gGiTraceConstants.rayMinDistance,
        max(gGiFrame.cameraNearFar.x, 1.0e-3f) * 1.0e-3f);
}

float giSceneRayMaxDistance()
{
    return min(gGiTraceConstants.rayMaxDistance,
        max(gGiFrame.cameraNearFar.y,
            gGiTraceConstants.cellSize * 256.0f));
}

struct GiSurfelData
{
    float3 position;
    float radius;
    float3 normal;
    float confidence;
    uint cellKey;
    uint materialId;
    uint lastSeenFrame;
    uint flags;
    uint instanceId;
    uint generation;
    uint spawnFrame;
    uint sceneGeneration;
    float meanLuma;
    float meanLumaSq;
    float sampleCount;
    uint lastQueuedFrame;
};

GiSurfelData giLoadSurfel(uint index)
{
    const uint a = index * GI_SURFEL_STRIDE;
    GiSurfelData s;
    const uint4 pr = gGiSurfels.Load4(a + 0u);
    const uint4 nc = gGiSurfels.Load4(a + 16u);
    const uint4 rv = gGiSurfels.Load4(a + 32u);
    const uint4 km = gGiSurfels.Load4(a + 48u);
    const uint4 ig = gGiSurfels.Load4(a + 64u);
    s.position = asfloat(pr.xyz); s.radius = asfloat(pr.w);
    s.normal = asfloat(nc.xyz); s.confidence = asfloat(nc.w);
    s.meanLuma = asfloat(rv.x); s.meanLumaSq = asfloat(rv.y);
    s.sampleCount = asfloat(rv.z); s.lastQueuedFrame = rv.w;
    s.cellKey = km.x; s.materialId = km.y; s.lastSeenFrame = km.z; s.flags = km.w;
    s.instanceId = ig.x; s.generation = ig.y;
    s.spawnFrame = ig.z; s.sceneGeneration = ig.w;
    return s;
}

// Appends a surfel to this frame's update queue exactly once (deduplicated by
// stamping the surfel's last-queued frame with an atomic exchange).
void giMarkTouched(uint surfelIndex)
{
    const GiSurfelData s = giLoadSurfel(surfelIndex);
    const float variance = max(s.meanLumaSq - s.meanLuma * s.meanLuma, 0.0f);
    const float standardErrorSq = variance / max(s.sampleCount, 1.0f);
    if (s.sampleCount >= 65536.0f ||
        (s.sampleCount >= 4096.0f && standardErrorSq < 1.0e-5f))
        return;
    // Do not let already useful surfels consume the bounded queue every frame.
    // This turns the old deterministic first-8192 starvation into a rotating,
    // confidence-driven stream while retaining more frequent high-variance work.
    uint revisitPeriod = 1u;
    if (s.confidence > 0.95f && standardErrorSq < 1.0e-4f)
        revisitPeriod = 32u;
    else if (s.confidence > 0.75f)
        revisitPeriod = 8u;
    if (s.lastQueuedFrame != 0u &&
        gGiTraceConstants.frameIndex - s.lastQueuedFrame < revisitPeriod)
        return;
    uint prev = 0u;
    gGiSurfels.InterlockedExchange(
        surfelIndex * GI_SURFEL_STRIDE + 44u,
        gGiTraceConstants.frameIndex, prev);
    if (prev == gGiTraceConstants.frameIndex) return;
    uint slot = 0u;
    gGiCounters.InterlockedAdd(4u, 1u, slot);   // counters[1] = touched count
    if (slot < gGiTraceConstants.maxUpdates)
        gGiAllocationQueue.Store(slot * 4u, surfelIndex);
    else
        gGiDiagnostics.InterlockedAdd(8u, 1u);  // allocationFailures
}

void giInitSurfel(uint index, uint cellKey, GiSurface surface,
    uint clipLevel, float levelCellSize)
{
    const uint a = index * GI_SURFEL_STRIDE;
    gGiSurfels.Store4(a + 0u,
        uint4(asuint(surface.position),
            asuint(levelCellSize * 0.75f)));
    gGiSurfels.Store4(a + 16u, uint4(asuint(surface.normal), asuint(0.0f)));
    gGiSurfels.Store4(a + 32u, uint4(0u, 0u, 0u, 0u));
    gGiSurfels.Store4(a + 48u,
        uint4(cellKey, surface.materialId, gGiTraceConstants.frameIndex,
            clipLevel & 7u));
    gGiSurfels.Store4(a + 64u,
        uint4(surface.instanceId, 0u, gGiTraceConstants.frameIndex,
            gGiTraceConstants.sceneGeneration));
    gGiSurfels.Store4(a + 80u, uint4(0u, 0u, 0u, 0u));
    gGiSurfels.Store4(a + 96u, uint4(0u, 0u, 0u, 0u));
    gGiSurfels.Store4(a + 112u, uint4(0u, 0u, 0u, 0u));
}

void giSurfelAllocate(uint3 id)
{
    if (gGiTraceConstants.freezeAfterFrame != 0u &&
        gGiTraceConstants.frameIndex >= gGiTraceConstants.freezeAfterFrame)
        return;
    const uint surfaceCount =
        gGiTraceConstants.reducedWidth * gGiTraceConstants.reducedHeight;
    if (id.x >= surfaceCount) return;
    const GiSurface surface = giLoadSurface(id.x);
    if (!surface.valid) return;

    const uint clipLevel = giSurfelClipLevel(surface.position);
    const float levelCellSize = giLevelCellSize(clipLevel);
    const int3 cell = giCellCoordAtLevel(surface.position, clipLevel);
    const uint cellKey = giCellKeyAtLevel(cell, clipLevel);
    const uint bucketCount = max(gGiTraceConstants.hashBucketCount, 1u);
    const uint bucketAddr = (cellKey % bucketCount) * 16u;
    const uint cands = clamp(gGiTraceConstants.candidatesPerCell, 1u, 4u);
    const uint maxSurfels = max(gGiTraceConstants.maxSurfels, 1u);

    // 1) Reuse a compatible surfel already resident in this cell.
    const uint4 slots = gGiHashBuckets.Load4(bucketAddr);
    [loop] for (uint k = 0u; k < cands; ++k)
    {
        const uint si = slots[k];
        if (si == GI_INVALID_INDEX || si >= maxSurfels) continue;
        const GiSurfelData s = giLoadSurfel(si);
        if (s.cellKey != cellKey ||
            s.sceneGeneration != gGiTraceConstants.sceneGeneration)
            continue;
        if ((s.flags & 7u) != clipLevel) continue;
        if (s.materialId != surface.materialId ||
            s.instanceId != surface.instanceId)
            continue;
        // A cell represents a finite patch, not one arbitrarily placed sample.
        // Reusing a distant coplanar surfel collapsed flat walls to one lighting
        // value per cell and produced stable but visibly blocky illumination.
        // Keep up to four spatially distinct samples in the existing slots.
        if (length(s.position - surface.position) >
                levelCellSize * 0.65f)
            continue;
        if (dot(normalize(s.normal), surface.normal) <
                gGiTraceConstants.normalThreshold)
            continue;
        const float allocationPlaneThreshold = max(
            gGiTraceConstants.planeThreshold,
            min(levelCellSize * 0.15f, 0.35f));
        if (abs(dot(s.position - surface.position, surface.normal)) >
                allocationPlaneThreshold)
            continue;
        gGiSurfels.Store(si * GI_SURFEL_STRIDE + 56u,
            gGiTraceConstants.frameIndex);  // keyMaterialAgeFlags.z = last seen
        giMarkTouched(si);
        return;
    }

    // 2) Atomically claim an empty or stale candidate. Never evict a live
    // colliding cell: doing so made fast backends continuously churn the cache
    // and destroy coverage. Multiple invocations may claim different slots for
    // the same cell, preserving the intended bounded multi-surfel representation.
    uint claimAddress = GI_INVALID_INDEX;
    uint claimExpected = GI_INVALID_INDEX;
    [loop] for (uint k = 0u; k < cands; ++k)
    {
        const uint si = slots[k];
        if (si == GI_RESERVED_INDEX) continue;
        bool claimable = si == GI_INVALID_INDEX || si >= maxSurfels;
        if (!claimable)
        {
            const GiSurfelData s = giLoadSurfel(si);
            claimable = s.sceneGeneration != gGiTraceConstants.sceneGeneration ||
                gGiTraceConstants.frameIndex - s.lastSeenFrame >
                    gGiTraceConstants.maxSurfelAge;
        }
        if (claimable)
        {
            claimAddress = bucketAddr + k * 4u;
            claimExpected = si;
            break;
        }
    }
    if (claimAddress == GI_INVALID_INDEX)
    {
        gGiDiagnostics.InterlockedAdd(4u, 1u); // live hash/candidate collision
        return;
    }

    uint observed = GI_INVALID_INDEX;
    gGiHashBuckets.InterlockedCompareExchange(
        claimAddress, claimExpected, GI_RESERVED_INDEX, observed);
    if (observed != claimExpected)
    {
        gGiDiagnostics.InterlockedAdd(4u, 1u);
        return;
    }

    // 3) Admit at most maxUpdates new surfels per frame. Surface preparation
    //    can expose hundreds of thousands of unique cells at once; allowing
    //    every pixel to initialize a 128-byte persistent record defeats the
    //    fixed update budget and can exceed the DX12 watchdog on the first
    //    cache population.  Rejected pixels are reconsidered next frame, so
    //    coverage grows progressively without CPU work or unbounded GPU work.
    uint allocationOrdinal = 0u;
    gGiCounters.InterlockedAdd(8u, 1u, allocationOrdinal);
    if (allocationOrdinal >= gGiTraceConstants.maxUpdates)
    {
        uint ignored = 0u;
        gGiHashBuckets.InterlockedCompareExchange(
            claimAddress, GI_RESERVED_INDEX, claimExpected, ignored);
        gGiDiagnostics.InterlockedAdd(8u, 1u);  // allocationFailures
        gGiDiagnostics.InterlockedOr(36u, 1u);  // overflowFlags: allocation budget
        return;
    }

    // Allocate from the persistent ring head (overflow-safe via modulo), then
    // publish only after the full surfel record is initialized.
    uint head = 0u;
    gGiResidualInterface.InterlockedAdd(0u, 1u, head);
    const uint newIndex = head % maxSurfels;
    giInitSurfel(newIndex, cellKey, surface, clipLevel, levelCellSize);
    gGiHashBuckets.Store(claimAddress, newIndex);
    giMarkTouched(newIndex);
}

void giPrioritize(uint3 id)
{
    // Touched surfels queued this frame, clamped to the update and ray budgets.
    const uint rays = max(gGiTraceConstants.raysPerSurfel, 1u);
    const uint maxAdaptiveRays = min(rays * 2u, 32u);
    const uint budgetCount = gGiTraceConstants.rayBudget / maxAdaptiveRays;
    const uint touched = gGiCounters.Load(4u); // counters[1]
    const bool frozen = gGiTraceConstants.freezeAfterFrame != 0u &&
        gGiTraceConstants.frameIndex >= gGiTraceConstants.freezeAfterFrame;
    const uint workCount = frozen ? 0u : min(touched,
        min(gGiTraceConstants.maxUpdates, budgetCount));
    if (id.x == 0u)
    {
        gGiIndirectArguments.Store3(0u,
            uint3(max((workCount + 63u) / 64u, 1u), 1u, 1u));
        gGiCounters.Store(0u, workCount); // work count consumed by trace/gather
    }
    if (id.x >= workCount) return;

    const uint surfelIndex = gGiAllocationQueue.Load(id.x * 4u);
    if (surfelIndex >= gGiTraceConstants.maxSurfels) return;
    const GiSurfelData s = giLoadSurfel(surfelIndex);

    // Adaptive ray count: invest more rays where the estimate is unconverged
    // (low confidence / high variance), fewer where it is stable.
    const float variance = max(s.meanLumaSq - s.meanLuma * s.meanLuma, 0.0f);
    uint adaptive = rays;
    if (s.confidence < 0.25f) adaptive = rays * 2u;
    else if (s.confidence > 0.9f && variance < 0.02f) adaptive = max(rays / 2u, 1u);
    adaptive = clamp(adaptive, 1u, 32u);

    // The sequence is a pure function of persistent cache identity. Camera
    // jitter, screen coordinates, and the frame on which a surfel happened to
    // become visible cannot rotate or restart its lighting estimate.
    const uint sampleBase = giHash(surfelIndex * 2654435761u ^
        gGiTraceConstants.sceneGeneration * 2246822519u);
    const uint sampleOffset = (uint)max(s.sampleCount, 0.0f);
    const float standardErrorSq = variance /
        max(s.sampleCount, 1.0f);
    // Keep high-variance records moving after the normal warm-up instead of
    // freezing a noisy cache merely because it reached a sample-count target.
    // The upper bound remains fixed and overflow-safe.
    if (sampleOffset >= 65536u ||
        (sampleOffset >= 4096u && standardErrorSq < 1.0e-5f))
        adaptive = 0u;

    const uint cmd = id.x * 32u;
    gGiPriorityQueue.Store4(cmd + 0u,
        uint4(surfelIndex, adaptive, sampleBase,
            gGiTraceConstants.sceneGeneration));
    gGiPriorityQueue.Store4(cmd + 16u,
        uint4(asuint(s.confidence), asuint(variance), s.materialId,
            sampleOffset));
    gGiDiagnostics.InterlockedAdd(28u, adaptive); // rayBudgetUsed
}

// L1 spherical-harmonic basis for a normalized direction.
float4 giShBasis(float3 d)
{
    return float4(0.282094792f, 0.488602512f * d.y,
        0.488602512f * d.z, 0.488602512f * d.x);
}

// Uniform hemisphere sample (pdf = 1/2pi) around n, decorrelated per surfel/frame.
float3 giUniformHemisphere(float3 n, uint seed, uint sampleIndex)
{
    const float u1 = frac(giRadicalInverse(sampleIndex) +
        float(giHash(seed)) * 2.3283064365e-10f);
    const float u2 = frac(giRadicalInverse(sampleIndex ^ 0x68bc21ebu) +
        float(giHash(seed + 0x9e3779b9u)) * 2.3283064365e-10f);
    const float z = u1;
    const float r = sqrt(max(0.0f, 1.0f - z * z));
    const float phi = 6.28318530718f * u2;
    const float3 local = float3(r * cos(phi), r * sin(phi), z);
    const float3 up = abs(n.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0);
    const float3 t = normalize(cross(up, n));
    const float3 b = cross(n, t);
    return normalize(t * local.x + b * local.y + n * local.z);
}

bool giVisible(float3 origin, float3 direction, float maxDistance)
{
    RayDesc shadowRay;
    shadowRay.Origin = origin;
    shadowRay.Direction = direction;
    const float sceneRayMin = giSceneRayMinDistance();
    shadowRay.TMin = sceneRayMin;
    shadowRay.TMax = max(maxDistance - sceneRayMin * 2.0f, sceneRayMin);
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadow;
    shadow.TraceRayInline(gGiSceneTlas,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xffu, shadowRay);
    [loop] while (shadow.Proceed())
    {
        if (shadow.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE)
            continue;
        const uint instanceId = shadow.CandidateInstanceID();
        if (instanceId >= gGiTraceConstants.instanceCount ||
            shadow.CandidateGeometryIndex() != 0u)
        {
            shadow.CommitNonOpaqueTriangleHit();
            continue;
        }
        const GiInstance instance = giLoadInstance(instanceId);
        if (instance.identity.x != instanceId ||
            instance.identity.z >= gGiTraceConstants.geometryCount)
        {
            shadow.CommitNonOpaqueTriangleHit();
            continue;
        }
        const GiGeometry geometry = giLoadGeometry(instance.identity.z);
        const uint primitiveIndex = shadow.CandidatePrimitiveIndex();
        const uint materialId = instance.state.z + geometry.mapping.y;
        if (geometry.mapping.x >= GI_MAX_MODEL_BUFFERS ||
            primitiveIndex * 3u + 2u >= geometry.range.w ||
            materialId >= gGiTraceConstants.materialCount)
        {
            shadow.CommitNonOpaqueTriangleHit();
            continue;
        }
        const uint3 indices = giLoadTriangleIndices(geometry, primitiveIndex);
        const GiVertex v0 = giLoadVertex(geometry.mapping.x, indices.x);
        const GiVertex v1 = giLoadVertex(geometry.mapping.x, indices.y);
        const GiVertex v2 = giLoadVertex(geometry.mapping.x, indices.z);
        const float2 uv = giBarycentricUv(v0, v1, v2,
            shadow.CandidateTriangleBarycentrics());
        const GiMaterial material = giLoadMaterial(materialId);
        float alpha = material.baseColor.a;
        if (giValidTextureSampler(
            material.baseColorTexture, material.baseColorSampler))
        {
            alpha *= giSampleTexture(material.baseColorTexture,
                material.baseColorSampler, uv).a;
        }
        if (alpha >= material.alphaCutoff)
            shadow.CommitNonOpaqueTriangleHit();
    }
    return shadow.CommittedStatus() == COMMITTED_NOTHING;
}

float3 giSampleEmissiveIrradiance(
    float3 position,
    float3 geometricNormal,
    float3 shadingNormal,
    uint seed,
    uint sampleIndex)
{
    const uint instanceId = gGiTraceConstants.emissiveInstanceIndex;
    if (instanceId == GI_INVALID_INDEX ||
        instanceId >= gGiTraceConstants.instanceCount)
        return 0.0f;
    const GiInstance instance = giLoadInstance(instanceId);
    if (instance.identity.x != instanceId ||
        instance.identity.z >= gGiTraceConstants.geometryCount)
        return 0.0f;
    const GiGeometry geometry = giLoadGeometry(instance.identity.z);
    const uint triangleCount = geometry.range.w / 3u;
    if (geometry.mapping.x >= GI_MAX_MODEL_BUFFERS || triangleCount == 0u)
        return 0.0f;

    const uint sequence = giHash(seed ^ sampleIndex ^ 0xa511e9b3u);
    const uint primitiveIndex = sequence % triangleCount;
    const uint3 indices = giLoadTriangleIndices(geometry, primitiveIndex);
    const GiVertex v0 = giLoadVertex(geometry.mapping.x, indices.x);
    const GiVertex v1 = giLoadVertex(geometry.mapping.x, indices.y);
    const GiVertex v2 = giLoadVertex(geometry.mapping.x, indices.z);
    const float xi0 = frac(float(giHash(sequence + 0x68bc21ebu)) *
        2.3283064365386963e-10f);
    const float xi1 = giRadicalInverse(sequence + 0x9e3779b9u);
    const float su = sqrt(xi0);
    const float3 bary = float3(1.0f - su, xi1 * su,
        (1.0f - xi1) * su);
    const float3 p0 = giTransformPoint(instance, v0.position);
    const float3 p1 = giTransformPoint(instance, v1.position);
    const float3 p2 = giTransformPoint(instance, v2.position);
    const float3 lightPoint = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    const float3 crossEdges = cross(p1 - p0, p2 - p0);
    const float area = 0.5f * length(crossEdges);
    if (area <= 1.0e-8f) return 0.0f;
    const float3 lightNormal = normalize(crossEdges);
    const float3 delta = lightPoint - position;
    const float distanceSq = max(dot(delta, delta), 1.0e-6f);
    const float distance = sqrt(distanceSq);
    const float3 lightDirection = delta / distance;
    const float nDotL = max(dot(shadingNormal, lightDirection), 0.0f);
    const uint materialId = instance.state.z + geometry.mapping.y;
    if (nDotL <= 0.0f || materialId >= gGiTraceConstants.materialCount)
        return 0.0f;
    const GiMaterial material = giLoadMaterial(materialId);
    const bool doubleSided =
        (material.flags & GI_MATERIAL_DOUBLE_SIDED) != 0u;
    const float lightCosine = doubleSided
        ? abs(dot(lightNormal, -lightDirection))
        : max(dot(lightNormal, -lightDirection), 0.0f);
    if (lightCosine <= 0.0f) return 0.0f;
    const float offset = giOriginOffset(position, geometricNormal, 1.0e-4f);
    if (!giVisible(position + geometricNormal * offset,
        lightDirection, distance))
        return 0.0f;

    const float2 uv = v0.uv * bary.x + v1.uv * bary.y + v2.uv * bary.z;
    float3 emission = max(material.emissive.rgb, 0.0f);
    if (giValidTextureSampler(
        material.emissiveTexture, material.emissiveSampler))
    {
        emission *= max(giSampleTexture(material.emissiveTexture,
            material.emissiveSampler, uv).rgb, 0.0f);
    }
    // Uniform triangle selection followed by uniform area sampling:
    // pdf = 1 / (triangleCount * triangleArea).
    return emission * nDotL * lightCosine *
        (float(triangleCount) * area / distanceSq);
}

float3 giSampleEmissiveIrradianceStratified(
    float3 position,
    float3 geometricNormal,
    float3 shadingNormal,
    uint seed,
    uint sampleIndex)
{
    float3 sum = 0.0f;
    [unroll] for (uint i = 0u; i < 4u; ++i)
    {
        sum += giSampleEmissiveIrradiance(position, geometricNormal,
            shadingNormal, seed ^ (i * 0x9e3779b9u),
            sampleIndex * 4u + i);
    }
    return sum * 0.25f;
}

// Visibility-shadowed direct irradiance from the shared scene lights and the
// first emissive primitive. Returns E (no hit albedo or Lambert 1/pi).
float3 giDirectIrradiance(
    float3 position,
    float3 geometricNormal,
    float3 shadingNormal,
    uint seed,
    uint sampleIndex)
{
    float3 e = 0.0f;
    const float3 sunDir = normalize(-gGiFrame.lightDirection);
    const float sunCosine = max(dot(shadingNormal, sunDir), 0.0f);
    const float originOffset = giOriginOffset(
        position, geometricNormal, 1.0e-4f);
    if (sunCosine > 0.0f && giVisible(
        position + geometricNormal * originOffset,
        sunDir, giSceneRayMaxDistance()))
    {
        e += gGiFrame.lightColor * gGiFrame.lightIntensity * sunCosine;
    }
    const uint count = min(gGiFrame.pointLightCount, GI_MAX_POINT_LIGHTS);
    [loop] for (uint i = 0u; i < count; ++i)
    {
        const float4 pr = gGiFrame.pointLightPositionRange[i];
        const float3 delta = pr.xyz - position;
        const float dist2 = dot(delta, delta);
        const float range = max(pr.w, 1.0e-3f);
        if (dist2 >= range * range) continue;
        const float dist = sqrt(max(dist2, 1.0e-8f));
        const float3 lightDirection = delta / dist;
        const float ndl = max(dot(shadingNormal, lightDirection), 0.0f);
        if (ndl <= 0.0f) continue;
        if (!giVisible(position + geometricNormal * originOffset,
            lightDirection, dist)) continue;
        float window = saturate(1.0f - dist2 / (range * range));
        window *= window;
        const float4 ci = gGiFrame.pointLightColorIntensity[i];
        e += ci.rgb * ci.a * ndl * window / max(dist2, 1.0e-4f);
    }
    e += giSampleEmissiveIrradianceStratified(position, geometricNormal,
        shadingNormal, seed, sampleIndex);
    return e;
}

static const float GI_PI = 3.14159265359f;
static const uint GI_PROBE_STRIDE = 128u;

uint giPositiveMod(int value, uint divisor)
{
    // Avoid signed remainder in the toroidal address calculation.  The Vulkan
    // path was effectively mapping negative cells through uint(value) % r, so
    // z=-2 and z=14 collided at resolution 24 and rewrote each other on adjacent
    // sweep segments.  Express the negative case entirely with unsigned math;
    // -(value + 1) is also safe for INT_MIN.
    if (value >= 0)
        return uint(value) % divisor;
    const uint magnitudeMinusOne = uint(-(value + 1));
    return divisor - 1u - (magnitudeMinusOne % divisor);
}

uint giProbeIndex(int3 absoluteCell, uint level)
{
    const uint resolution = giProbeResolution();
    const uint3 wrapped = uint3(
        giPositiveMod(absoluteCell.x, resolution),
        giPositiveMod(absoluteCell.y, resolution),
        giPositiveMod(absoluteCell.z, resolution));
    const uint probesPerLevel = resolution * resolution * resolution;
    return level * probesPerLevel + wrapped.x +
        resolution * (wrapped.y + resolution * wrapped.z);
}

float giProbeSpacing(uint level)
{
    return giLevelCellSize(level) * 4.0f;
}

uint giProbeSweepLength()
{
    const uint resolution = giProbeResolution();
    const uint verticalSlices = min(resolution, 8u);
    const uint total = resolution * resolution * verticalSlices *
        giClipmapCount();
    const uint budget = max(
        (uint)round(gGiTraceConstants.confidenceBlend), 1u);
    return max((total + budget - 1u) / budget, 1u);
}

uint giProbeRelativeFrame()
{
    return gGiTraceConstants.frameIndex - gGiResidualInterface.Load(8u);
}

bool giProbeSourceIsStaging()
{
    return (giProbeRelativeFrame() & 1u) != 0u;
}

bool giProbeReadIsStaging()
{
#if defined(IC_GI_READ_SOURCE_FIELD)
    return giProbeSourceIsStaging();
#else
    // PreserveProbes completes the opposite field, UpdateProbes modifies only
    // that preserved destination, and all consumers later in the frame read it.
    return !giProbeSourceIsStaging();
#endif
}

uint4 giLoadProbe4(bool sourceIsStaging, uint address)
{
    return sourceIsStaging
        ? gGiProbeStaging.Load4(address)
        : gGiProbes.Load4(address);
}

void giStoreProbeDestination4(
    bool sourceIsStaging, uint address, uint4 value)
{
    if (sourceIsStaging)
        gGiProbes.Store4(address, value);
    else
        gGiProbeStaging.Store4(address, value);
}

void giPreserveProbes(uint3 id)
{
    const uint resolution = giProbeResolution();
    const uint probeCount = resolution * resolution * resolution *
        giClipmapCount();
    if (id.x >= probeCount) return;
    const bool sourceIsStaging = giProbeSourceIsStaging();
    const uint address = id.x * GI_PROBE_STRIDE;
    [unroll] for (uint offset = 0u; offset < GI_PROBE_STRIDE; offset += 16u)
    {
        giStoreProbeDestination4(sourceIsStaging, address + offset,
            giLoadProbe4(sourceIsStaging, address + offset));
    }
}

bool giSurfelDetailEnabled()
{
    return (gGiTraceConstants.giFlags & (1u << 21u)) != 0u;
}

float3 giProbeSphereDirection(uint probeKey, uint sampleIndex, uint sampleCount)
{
    // A world probe always uses the same stratified directions. Rotating the
    // set every update continuously injected Monte-Carlo noise into otherwise
    // static lighting, which was most visible as dark-area sparkle.
    const uint ordinal = sampleIndex % sampleCount;
    const float u = (float(ordinal) + 0.5f) / float(sampleCount);
    const float z = 1.0f - 2.0f * u;
    const float radius = sqrt(max(1.0f - z * z, 0.0f));
    const float rotation = float(giHash(probeKey)) *
        2.3283064365386963e-10f * 6.28318530718f;
    const float phi = 2.39996322973f * float(ordinal) + rotation;
    return float3(radius * cos(phi), radius * sin(phi), z);
}

float3 giIrradianceFromSH(float4 shR, float4 shG, float4 shB, float3 n);

struct GiProbeLevelSample
{
    float3 irradiance;
    float confidence;
    float visibility;
    float variance;
    float age;
    float relocation;
    float3 rawRadiance;
    uint selectedIndex;
};

GiProbeLevelSample giGatherProbeLevel(
    float3 position, float3 normal, uint level)
{
    GiProbeLevelSample result;
    result.irradiance = 0.0f;
    result.confidence = 0.0f;
    result.visibility = 0.0f;
    result.variance = 0.0f;
    result.age = 0.0f;
    result.relocation = 0.0f;
    result.rawRadiance = 0.0f;
    result.selectedIndex = GI_INVALID_INDEX;

    const float spacing = giProbeSpacing(level);
    const float3 gridPosition = position / spacing - 0.5f;
    const int3 baseCell = int3(floor(gridPosition));
    const float3 trilinear = frac(gridPosition);
    const bool updatesFrozen = gGiTraceConstants.freezeAfterFrame != 0u &&
        gGiTraceConstants.frameIndex >= gGiTraceConstants.freezeAfterFrame;
    const bool sourceIsStaging = giProbeReadIsStaging();
    float3 sum = 0.0f;
    float weight = 0.0f;
    float visibilitySum = 0.0f;
    float varianceSum = 0.0f;
    float ageSum = 0.0f;
    float relocationSum = 0.0f;
    float3 rawRadianceSum = 0.0f;
    float selectedWeight = 0.0f;
    // Coverage (probe support) is tracked separately from the irradiance weight.
    // The irradiance weight is heavily attenuated by soft occlusion and variance
    // so that darkened/occluded probes still shade correctly; using that same
    // value as coverage made well-supported surfaces read as low-coverage and
    // fall back to flat, contrast-free IBL. Coverage instead accumulates the
    // front-facing trilinear support of converged probes (occluded probes still
    // count, so contact/corner areas keep their darkened GI rather than washing
    // out to ambient).
    float coverageWeight = 0.0f;
    [unroll] for (int oz = 0; oz <= 1; ++oz)
    [unroll] for (int oy = 0; oy <= 1; ++oy)
    [unroll] for (int ox = 0; ox <= 1; ++ox)
    {
        const int3 cell = baseCell + int3(ox, oy, oz);
        const uint probeIndex = giProbeIndex(cell, level);
        const uint address = probeIndex * GI_PROBE_STRIDE;
        const uint4 flags = giLoadProbe4(sourceIsStaging, address + 112u);
        if ((flags.w & 1u) == 0u ||
            flags.x != giCellKeyAtLevel(cell, level) || flags.y != level ||
            (!updatesFrozen && gGiTraceConstants.frameIndex - flags.z >
                gGiTraceConstants.maxSurfelAge * 2u))
            continue;
        const float4 positionConfidence = asfloat(
            giLoadProbe4(sourceIsStaging, address + 96u));
        if (positionConfidence.w <= 0.0f) continue;
        const float3 biasedPosition = position + normal *
            max(0.04f * spacing, gGiTraceConstants.rayMinDistance * 4.0f);
        const float3 delta = biasedPosition - positionConfidence.xyz;
        const float distanceToProbe = length(delta);
        if (distanceToProbe > spacing * 2.25f) continue;
        const float3 direction = delta / max(distanceToProbe, 1.0e-4f);
        const float4 axis0 = asfloat(
            giLoadProbe4(sourceIsStaging, address + 64u));
        const float4 axis1 = asfloat(
            giLoadProbe4(sourceIsStaging, address + 80u));
        const float4 axisWeight0 = float4(max(direction.x, 0.0f),
            max(-direction.x, 0.0f), max(direction.y, 0.0f),
            max(-direction.y, 0.0f));
        const float2 axisWeight1 = float2(max(direction.z, 0.0f),
            max(-direction.z, 0.0f));
        const float predictedDepth =
            (dot(axis0, axisWeight0) + dot(axis1.xy, axisWeight1)) /
            max(dot(axisWeight0, 1.0f.xxxx) +
                dot(axisWeight1, 1.0f.xx), 1.0e-4f);
        const float4 depthMoments = asfloat(
            giLoadProbe4(sourceIsStaging, address + 48u));
        const float deltaDepth = max(distanceToProbe -
            predictedDepth - 0.06f * spacing, 0.0f);
        const float visibility = deltaDepth <= 0.0f ? 1.0f :
            saturate(depthMoments.y /
                (depthMoments.y + deltaDepth * deltaDepth));
        const float wx = ox == 0 ? 1.0f - trilinear.x : trilinear.x;
        const float wy = oy == 0 ? 1.0f - trilinear.y : trilinear.y;
        const float wz = oz == 0 ? 1.0f - trilinear.z : trilinear.z;
        const float trilinearWeight = wx * wy * wz;
        const float varianceWeight = rcp(1.0f +
            sqrt(max(depthMoments.w, 0.0f)));
        // Suppress probes on the far side of the receiving surface. This is a
        // continuous weight, so it strengthens indirect occlusion without the
        // popping caused by a binary normal test.
        const float facing = saturate(dot(normal, -direction) * 0.5f + 0.5f);
        coverageWeight += trilinearWeight * facing * positionConfidence.w;
        const float w = trilinearWeight * visibility * visibility *
            facing * facing * positionConfidence.w * varianceWeight;
        if (w <= 0.0f) continue;
        const float4 shR = asfloat(
            giLoadProbe4(sourceIsStaging, address + 0u));
        const float4 shG = asfloat(
            giLoadProbe4(sourceIsStaging, address + 16u));
        const float4 shB = asfloat(
            giLoadProbe4(sourceIsStaging, address + 32u));
        sum += giIrradianceFromSH(shR, shG, shB, normal) * w;
        weight += w;
        visibilitySum += visibility * w;
        varianceSum += max(depthMoments.w, 0.0f) * w;
        ageSum += min(float(gGiTraceConstants.frameIndex - flags.z),
            1024.0f) * w;
        const float3 nominalPosition = (float3(cell) + 0.5f) * spacing;
        relocationSum += length(positionConfidence.xyz - nominalPosition) /
            max(spacing, 1.0e-4f) * w;
        rawRadianceSum += max(float3(shR.x, shG.x, shB.x) *
            0.282094792f, 0.0f) * w;
        if (w > selectedWeight)
        {
            selectedWeight = w;
            result.selectedIndex = probeIndex;
        }
    }
    if (weight <= 1.0e-5f) return result;
    result.irradiance = sum / weight;
    result.confidence = saturate(coverageWeight);
    result.visibility = visibilitySum / weight;
    result.variance = varianceSum / weight;
    result.age = ageSum / weight;
    result.relocation = relocationSum / weight;
    result.rawRadiance = rawRadianceSum / weight;
    return result;
}

GiProbeLevelSample giBlendProbeSamples(
    GiProbeLevelSample a, GiProbeLevelSample b, float t)
{
    if (a.confidence <= 0.0f) return b;
    if (b.confidence <= 0.0f) return a;
    GiProbeLevelSample result;
    result.irradiance = lerp(a.irradiance, b.irradiance, t);
    result.confidence = lerp(a.confidence, b.confidence, t);
    result.visibility = lerp(a.visibility, b.visibility, t);
    result.variance = lerp(a.variance, b.variance, t);
    result.age = lerp(a.age, b.age, t);
    result.relocation = lerp(a.relocation, b.relocation, t);
    result.rawRadiance = lerp(a.rawRadiance, b.rawRadiance, t);
    result.selectedIndex = t < 0.5f ? a.selectedIndex : b.selectedIndex;
    return result;
}

float3 giSampleProbeIrradianceDetailed(float3 position, float3 normal,
    out float confidence, out GiProbeLevelSample debugSample)
{
    confidence = 0.0f;
    debugSample = (GiProbeLevelSample)0;
    debugSample.selectedIndex = GI_INVALID_INDEX;
    if (!giProbeFallbackEnabled()) return 0.0f;
    const uint levelCount = giClipmapCount();
    const float nearExtent = max(gGiTraceConstants.cellSize *
        float(giProbeResolution()) * 1.6f, 1.0e-3f);
    const float continuousLevel = clamp(log2(max(length(position -
        gGiFrame.cameraPosition) / nearExtent, 1.0f)), 0.0f,
        float(levelCount - 1u));
    const uint baseLevel = min((uint)floor(continuousLevel), levelCount - 1u);
    const uint nextLevel = min(baseLevel + 1u, levelCount - 1u);
    const float transition = baseLevel == nextLevel ? 0.0f :
        smoothstep(0.15f, 0.85f, frac(continuousLevel));
    const GiProbeLevelSample baseSample = giGatherProbeLevel(
        position, normal, baseLevel);
    GiProbeLevelSample nextSample = baseSample;
    if (nextLevel != baseLevel)
        nextSample = giGatherProbeLevel(position, normal, nextLevel);
    debugSample = giBlendProbeSamples(baseSample, nextSample, transition);
    // If both transition levels are empty, walk outward to the first valid
    // coarse level. This is a coverage fallback, never an abrupt fine/coarse
    // switch when both levels contain valid data.
    if (debugSample.confidence <= 0.0f)
    {
        [loop] for (uint level = nextLevel + 1u; level < levelCount; ++level)
        {
            debugSample = giGatherProbeLevel(position, normal, level);
            if (debugSample.confidence > 0.0f) break;
        }
    }
    confidence = debugSample.confidence;
    return debugSample.irradiance;
}

float3 giSampleProbeIrradiance(float3 position, float3 normal,
    out float confidence)
{
    GiProbeLevelSample debugSample;
    return giSampleProbeIrradianceDetailed(
        position, normal, confidence, debugSample);
}

void giTraceSurfelUpdates(uint3 id)
{
    const uint workCount = gGiCounters.Load(0u);
    if (id.x >= workCount) return;
    const uint commandAddress = id.x * 32u;
    const uint4 command0 = gGiPriorityQueue.Load4(commandAddress + 0u);
    const uint4 command1 = gGiPriorityQueue.Load4(commandAddress + 16u);
    const uint cacheIndex = command0.x;
    if (command0.y == 0u) return;
    const uint rayCount = min(command0.y, 32u);
    const uint sampleBase = command0.z;
    const uint sampleOffset = command1.w;
    const uint queuedSceneGeneration = command0.w;
    if (cacheIndex >= gGiTraceConstants.maxSurfels)
    {
        gGiDiagnostics.InterlockedAdd(68u, 1u);
        return;
    }
    const uint surfelAddress = cacheIndex * GI_SURFEL_STRIDE;
    const GiSurfelData surfel = giLoadSurfel(cacheIndex);
    // Reject stale surfels (scene rebuilt since it was queued/spawned).
    if (surfel.sceneGeneration != gGiTraceConstants.sceneGeneration ||
        queuedSceneGeneration != gGiTraceConstants.sceneGeneration)
    {
        gGiDiagnostics.InterlockedAdd(64u, 1u); // staleSurfels
        return;
    }

    const float3 sourcePosition = surfel.position;
    const float3 sourceNormal = normalize(surfel.normal);
    const float sourceError = 1.0e-4f * (1.0f + length(sourcePosition));
    const float originOffset = giOriginOffset(
        sourcePosition, sourceNormal, sourceError);

    // Per-frame radiance SH estimate (L1, one float3 per basis coefficient).
    float3 c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float lumaSum = 0.0f;
    float lumaSquareSum = 0.0f;

    float distanceSum = 0.0f;
    float distanceSquareSum = 0.0f;
    float visibilitySum = 0.0f;
    float3 visibilityDirection = 0.0f;
    float3 radianceSum = 0.0f;
    float3 referenceIrradianceSum = 0.0f;
    float3 sourceEmissiveIrradianceSum = 0.0f;
    float closestDistance = giSceneRayMaxDistance();
    float3 closestPosition = 0.0f;
    float3 closestGeometricNormal = 0.0f;
    float3 closestShadingNormal = 0.0f;
    float2 closestUv = 0.0f;
    float2 closestBary = 0.0f;
    uint4 closestIndices = uint4(GI_INVALID_INDEX, GI_INVALID_INDEX,
        GI_INVALID_INDEX, GI_INVALID_INDEX);
    uint closestGeneration = 0u;
    uint closestFlags = 0u;
    bool closestFrontFace = true;
    const GiMaterial sourceMaterial = giLoadMaterial(surfel.materialId);
    const bool sourceIsEmissive = dot(sourceMaterial.emissive.rgb,
        sourceMaterial.emissive.rgb) > 0.0f;

    [loop] for (uint rayIndex = 0u; rayIndex < rayCount; ++rayIndex)
    {
        const float3 direction = giUniformHemisphere(
            sourceNormal, sampleBase, sampleOffset + rayIndex);
        RayDesc ray;
        ray.Origin = sourcePosition + sourceNormal * originOffset;
        ray.Direction = direction;
        ray.TMin = giSceneRayMinDistance();
        ray.TMax = giSceneRayMaxDistance();
        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(gGiSceneTlas, RAY_FLAG_NONE, 0xffu, ray);
        [loop] while (query.Proceed())
        {
            if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE)
                continue;
            const uint candidateInstanceId = query.CandidateInstanceID();
            if (candidateInstanceId >= gGiTraceConstants.instanceCount ||
                query.CandidateGeometryIndex() != 0u)
            {
                gGiDiagnostics.InterlockedAdd(68u, 1u);
                continue;
            }
            const GiInstance candidateInstance = giLoadInstance(candidateInstanceId);
            if (candidateInstance.identity.x != candidateInstanceId ||
                candidateInstance.identity.z >= gGiTraceConstants.geometryCount)
            {
                gGiDiagnostics.InterlockedAdd(68u, 1u);
                continue;
            }
            const GiGeometry candidateGeometry =
                giLoadGeometry(candidateInstance.identity.z);
            const uint primitiveIndex = query.CandidatePrimitiveIndex();
            if (candidateGeometry.mapping.x >= GI_MAX_MODEL_BUFFERS ||
                primitiveIndex * 3u + 2u >= candidateGeometry.range.w)
            {
                gGiDiagnostics.InterlockedAdd(68u, 1u);
                continue;
            }
            bool accepted = true;
            if ((candidateGeometry.mapping.z & GI_GEOMETRY_ALPHA_TESTED) != 0u)
            {
                const uint materialId = candidateInstance.state.z +
                    candidateGeometry.mapping.y;
                if (materialId >= gGiTraceConstants.materialCount)
                {
                    gGiDiagnostics.InterlockedAdd(68u, 1u);
                    accepted = false;
                }
                else
                {
                    const uint3 triIndices = giLoadTriangleIndices(
                        candidateGeometry, primitiveIndex);
                    const GiVertex v0 = giLoadVertex(candidateGeometry.mapping.x, triIndices.x);
                    const GiVertex v1 = giLoadVertex(candidateGeometry.mapping.x, triIndices.y);
                    const GiVertex v2 = giLoadVertex(candidateGeometry.mapping.x, triIndices.z);
                    const float2 uv = giBarycentricUv(v0, v1, v2,
                        query.CandidateTriangleBarycentrics());
                    const GiMaterial material = giLoadMaterial(materialId);
                    float alpha = material.baseColor.a;
                    if (giValidTextureSampler(
                        material.baseColorTexture, material.baseColorSampler))
                        alpha *= giSampleTexture(material.baseColorTexture,
                            material.baseColorSampler, uv).a;
                    accepted = alpha >= material.alphaCutoff;
                }
            }
            if (accepted)
                query.CommitNonOpaqueTriangleHit();
            else
                gGiDiagnostics.InterlockedAdd(60u, 1u);
        }

        gGiDiagnostics.InterlockedAdd(48u, 1u);
        float distance = giSceneRayMaxDistance();
        float3 rayRadiance = 0.0f;
        bool hitEmissive = false;
        bool hit = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
        if (hit)
        {
            const uint hitInstanceId = query.CommittedInstanceID();
            if (hitInstanceId >= gGiTraceConstants.instanceCount ||
                query.CommittedGeometryIndex() != 0u)
            {
                gGiDiagnostics.InterlockedAdd(68u, 1u);
                hit = false;
            }
            else
            {
                const GiInstance hitInstance = giLoadInstance(hitInstanceId);
                const uint geometryIndex = hitInstance.identity.z;
                if (hitInstance.identity.x != hitInstanceId ||
                    geometryIndex >= gGiTraceConstants.geometryCount)
                {
                    gGiDiagnostics.InterlockedAdd(68u, 1u);
                    hit = false;
                }
                else
                {
                    const GiGeometry geometry = giLoadGeometry(geometryIndex);
                    const uint primitiveIndex = query.CommittedPrimitiveIndex();
                    const float2 bary = query.CommittedTriangleBarycentrics();
                    if (geometry.mapping.x >= GI_MAX_MODEL_BUFFERS ||
                        primitiveIndex * 3u + 2u >= geometry.range.w)
                    {
                        gGiDiagnostics.InterlockedAdd(68u, 1u);
                        hit = false;
                    }
                    else
                    {
                        const uint3 triIndices = giLoadTriangleIndices(geometry, primitiveIndex);
                        const GiVertex v0 = giLoadVertex(geometry.mapping.x, triIndices.x);
                        const GiVertex v1 = giLoadVertex(geometry.mapping.x, triIndices.y);
                        const GiVertex v2 = giLoadVertex(geometry.mapping.x, triIndices.z);
                        const float w = 1.0f - bary.x - bary.y;
                        const float3 objectPosition = v0.position * w +
                            v1.position * bary.x + v2.position * bary.y;
                        const float3 worldPosition = giTransformPoint(hitInstance, objectPosition);
                        const float3 wp0 = giTransformPoint(hitInstance, v0.position);
                        const float3 wp1 = giTransformPoint(hitInstance, v1.position);
                        const float3 wp2 = giTransformPoint(hitInstance, v2.position);
                        float3 geometricNormal = normalize(cross(wp1 - wp0, wp2 - wp0));
                        float3 shadingNormal = giTransformNormal(hitInstance,
                            normalize(v0.normal * w + v1.normal * bary.x + v2.normal * bary.y));
                        const float2 uv = giBarycentricUv(v0, v1, v2, bary);
                        const uint materialId = hitInstance.state.z + geometry.mapping.y;
                        if (materialId >= gGiTraceConstants.materialCount)
                        {
                            gGiDiagnostics.InterlockedAdd(68u, 1u);
                            hit = false;
                        }
                        else
                        {
                            const GiMaterial material = giLoadMaterial(materialId);
                            const float4 tangent = v0.tangent * w +
                                v1.tangent * bary.x + v2.tangent * bary.y;
                            if (giValidTextureSampler(material.normalTexture,
                                material.normalSampler))
                            {
                                const float3 normalSample = giSampleTexture(
                                    material.normalTexture, material.normalSampler, uv).xyz * 2.0f - 1.0f;
                                const float3 transformedTangent =
                                    giTransformDirection(hitInstance, tangent.xyz);
                                const float3 worldTangent = normalize(
                                    transformedTangent - shadingNormal *
                                    dot(transformedTangent, shadingNormal));
                                const float3 worldBitangent = normalize(
                                    cross(shadingNormal, worldTangent) * tangent.w);
                                shadingNormal = normalize(worldTangent * normalSample.x +
                                    worldBitangent * normalSample.y + shadingNormal * normalSample.z);
                            }
                            const bool frontFace = query.CommittedTriangleFrontFace();
                            if (dot(geometricNormal, direction) > 0.0f)
                                geometricNormal = -geometricNormal;
                            if (dot(shadingNormal, direction) > 0.0f)
                                shadingNormal = -shadingNormal;
                            if (dot(shadingNormal, geometricNormal) < 0.0f)
                                shadingNormal = -shadingNormal;
                            distance = query.CommittedRayT();
                            const float3 rayPosition = ray.Origin + direction * distance;
                            const float reconstructionError = length(worldPosition - rayPosition);
                            // Instance-agnostic self-intersection rejection: any hit
                            // within a small multiple of the origin offset is the
                            // surfel's own surface. The threshold is microns-to-mm so
                            // legitimate thin-wall geometry is preserved.
                            const bool selfHit = distance <= originOffset * 2.0f;
                            if (selfHit)
                            {
                                gGiDiagnostics.InterlockedAdd(72u, 1u);
                                hit = false;
                            }
                            else
                            {
                                float3 emissive = material.emissive.rgb;
                                if (giValidTextureSampler(material.emissiveTexture,
                                    material.emissiveSampler))
                                    emissive *= giSampleTexture(material.emissiveTexture,
                                        material.emissiveSampler, uv).rgb;
                                hitEmissive = dot(emissive, emissive) > 0.0f;
                                float3 albedo = material.baseColor.rgb;
                                if (giValidTextureSampler(material.baseColorTexture,
                                    material.baseColorSampler))
                                    albedo *= giSampleTexture(material.baseColorTexture,
                                        material.baseColorSampler, uv).rgb;
                                float metallic = saturate(material.metallicFactor);
                                if (giValidTextureSampler(
                                    material.metallicRoughnessTexture,
                                    material.metallicRoughnessSampler))
                                {
                                    metallic *= giSampleTexture(
                                        material.metallicRoughnessTexture,
                                        material.metallicRoughnessSampler, uv).b;
                                }
                                metallic = saturate(metallic);
                                float ao = 1.0f;
                                if (giValidTextureSampler(material.occlusionTexture,
                                    material.occlusionSampler))
                                {
                                    const float sampledAo = giSampleTexture(
                                        material.occlusionTexture,
                                        material.occlusionSampler, uv).r;
                                    ao = lerp(1.0f, sampledAo,
                                        saturate(material.occlusionStrength));
                                }
                                // One diffuse bounce. The cache stores incident
                                // radiance SH; only the secondary material is
                                // applied here. The receiver albedo is applied
                                // exactly once during forward injection.
                                const float3 directE = giDirectIrradiance(
                                    worldPosition, geometricNormal, shadingNormal,
                                    sampleBase, sampleOffset + rayIndex);
                                float feedbackConfidence = 0.0f;
                                float3 feedbackE = 0.0f;
                                if (gGiTraceConstants.feedbackEnabled != 0u)
                                {
                                    feedbackE = giSampleProbeIrradiance(
                                        worldPosition, shadingNormal,
                                        feedbackConfidence);
                                    // A strict contraction bound prevents a
                                    // high-albedo cache loop from amplifying its
                                    // own approximation error while preserving
                                    // energy-conserving diffuse feedback.
                                    feedbackConfidence = min(
                                        feedbackConfidence, 0.9f);
                                    if (feedbackConfidence > 0.0f)
                                        gGiDiagnostics.InterlockedAdd(120u, 1u);
                                }
                                rayRadiance = emissive +
                                    max(albedo, 0.0f) * ao * (1.0f - metallic) *
                                    (1.0f / GI_PI) *
                                    (directE + feedbackE * feedbackConfidence);
                                if (distance < closestDistance)
                                {
                                    closestDistance = distance;
                                    closestPosition = worldPosition;
                                    closestGeometricNormal = geometricNormal;
                                    closestShadingNormal = shadingNormal;
                                    closestUv = uv;
                                    closestBary = bary;
                                    closestIndices = uint4(hitInstanceId,
                                        geometryIndex, primitiveIndex, materialId);
                                    closestGeneration = hitInstance.identity.y;
                                    closestFlags = geometry.mapping.z;
                                    closestFrontFace = frontFace;
                                    const uint record = cacheIndex * GI_HIT_RECORD_STRIDE;
                                    gGiHitRecords.Store4(record + 96u,
                                        uint4(asuint(bary), asuint(reconstructionError), 0u));
                                }
                            }
                        }
                    }
                }
            }
        }
        if (hit)
        {
            gGiDiagnostics.InterlockedAdd(52u, 1u);
            visibilitySum += 1.0f;
            visibilityDirection += direction;
        }
        else
        {
            gGiDiagnostics.InterlockedAdd(56u, 1u);
            distance = giSceneRayMaxDistance();
            // Environment radiance belongs on unoccluded ray misses. Forward
            // injection replaces (rather than adds to) diffuse IBL according to
            // cache coverage, so this path is represented exactly once.
            rayRadiance = gGiTraceConstants.environmentEnabled != 0u
                ? max(gGiEnvironment.SampleLevel(
                    gGiEnvironmentSampler, direction, 0.0f).rgb, 0.0f) *
                    (gGiTraceConstants.environmentIntensity *
                        gGiFrame.environmentTransportExposure)
                : 0.0f;
        }
        distanceSum += distance;
        distanceSquareSum += distance * distance;
        radianceSum += rayRadiance;
        const float sourceCosine =
            max(dot(sourceNormal, direction), 0.0f);
        // Direct emitter visibility is estimated separately with NEE below;
        // excluding emissive hits here removes the rare bright-hit estimator
        // without changing the represented light paths.
        referenceIrradianceSum +=
            (hitEmissive ? 0.0f : rayRadiance) * sourceCosine;
        if (!sourceIsEmissive)
        {
            sourceEmissiveIrradianceSum +=
                giSampleEmissiveIrradianceStratified(sourcePosition,
                    sourceNormal, sourceNormal,
                    sampleBase ^ 0x51ed270bu,
                    sampleOffset + rayIndex);
        }
        const float4 Y = giShBasis(direction);
        c0 += rayRadiance * Y.x;
        c1 += rayRadiance * Y.y;
        c2 += rayRadiance * Y.z;
        c3 += rayRadiance * Y.w;
        const float irradianceSampleLuma = dot(
            rayRadiance * sourceCosine * 6.28318530718f,
            float3(0.2126f, 0.7152f, 0.0722f));
        lumaSum += irradianceSampleLuma;
        lumaSquareSum += irradianceSampleLuma * irradianceSampleLuma;
    }

    const float invRayCount = rcp(float(rayCount));
    const float meanDistance = distanceSum * invRayCount;
    const float meanSquareDistance = distanceSquareSum * invRayCount;
    const float hitRatio = visibilitySum * invRayCount;

    // This frame's radiance-SH estimate. Uniform hemisphere pdf = 1/2pi, so the
    // Monte Carlo projection scales each coefficient by 2pi / rayCount.
    const float shScale = 6.28318530718f * invRayCount;
    const float4 frameR = float4(c0.x, c1.x, c2.x, c3.x) * shScale;
    const float4 frameG = float4(c0.y, c1.y, c2.y, c3.y) * shScale;
    const float4 frameB = float4(c0.z, c1.z, c2.z, c3.z) * shScale;

    // Ray-count-weighted progressive average. A fixed blend floor keeps a
    // converged cache permanently stochastic, so accumulation is based solely
    // on the number of samples represented by each estimate.
    const float storedSamples = surfel.sampleCount;
    const float newSamples = storedSamples + float(rayCount);
    const float blend = float(rayCount) / max(newSamples, 1.0f);
    const float4 oldR = asfloat(gGiSurfels.Load4(surfelAddress + 80u));
    const float4 oldG = asfloat(gGiSurfels.Load4(surfelAddress + 96u));
    const float4 oldB = asfloat(gGiSurfels.Load4(surfelAddress + 112u));
    const float4 newR = lerp(oldR, frameR, blend);
    const float4 newG = lerp(oldG, frameG, blend);
    const float4 newB = lerp(oldB, frameB, blend);
    gGiSurfels.Store4(surfelAddress + 80u, asuint(newR));
    gGiSurfels.Store4(surfelAddress + 96u, asuint(newG));
    gGiSurfels.Store4(surfelAddress + 112u, asuint(newB));

    // Bounded brute-force reference: cosine-weight the same one-bounce ray
    // radiance directly, without SH encoding or reconstruction.
    const float3 referenceIrradiance = referenceIrradianceSum *
        (6.28318530718f * invRayCount) +
        sourceEmissiveIrradianceSum * invRayCount;
    const float3 oldReferenceIrradiance = asfloat(
        gGiVisibilityMoments.Load3(cacheIndex * 32u + 16u));
    const float3 accumulatedReferenceIrradiance = storedSamples > 0.0f
        ? lerp(oldReferenceIrradiance, referenceIrradiance, blend)
        : referenceIrradiance;
    const float4 shWeights = float4(GI_PI, 2.09439510f, 2.09439510f,
        2.09439510f) * giShBasis(sourceNormal);
    const float3 reconstructedIrradiance = max(float3(
        dot(frameR, shWeights), dot(frameG, shWeights),
        dot(frameB, shWeights)), 0.0f);
    const float referenceError = abs(dot(referenceIrradiance -
        reconstructedIrradiance, float3(0.2126f, 0.7152f, 0.0722f)));
    gGiDiagnostics.InterlockedAdd(88u,
        (uint)round(min(referenceError, 16.0f) * 4096.0f));
    gGiDiagnostics.InterlockedAdd(92u, 1u);

    const float frameLuma = lumaSum * invRayCount;
    const float frameLumaSq = lumaSquareSum * invRayCount;
    const float newMeanLuma = lerp(surfel.meanLuma, frameLuma, blend);
    const float newMeanLumaSq =
        lerp(surfel.meanLumaSq, frameLumaSq, blend);
    const float estimatorVariance = max(
        newMeanLumaSq - newMeanLuma * newMeanLuma, 0.0f);
    const float standardError = sqrt(estimatorVariance /
        max(newSamples, 1.0f));
    const float confidence = saturate(newSamples / 256.0f) /
        (1.0f + standardError * 4.0f);
    gGiSurfels.Store4(surfelAddress + 32u,
        uint4(asuint(newMeanLuma), asuint(newMeanLumaSq),
            asuint(newSamples), gGiTraceConstants.frameIndex));
    gGiSurfels.Store(surfelAddress + 28u, asuint(confidence));

    // Directional visibility moments plus the exact cosine-weighted irradiance
    // control value. SH remains available for differently oriented receivers;
    // the control value avoids discarding the higher-order visibility signal at
    // the surfel's own normal.
    gGiVisibilityMoments.Store4(cacheIndex * 32u + 0u,
        uint4(asuint(meanDistance), asuint(meanSquareDistance), asuint(hitRatio),
            asuint(max(0.0f, meanSquareDistance - meanDistance * meanDistance))));
    gGiVisibilityMoments.Store4(cacheIndex * 32u + 16u,
        uint4(asuint(accumulatedReferenceIrradiance), asuint(newSamples)));

    gGiDiagnostics.InterlockedAdd(12u, 1u);              // surfelUpdateCount
    gGiDiagnostics.InterlockedMax(32u, asuint(confidence)); // confidence proxy

    // Compact closest-hit record (debug views only).
    const uint record = cacheIndex * GI_HIT_RECORD_STRIDE;
    const float3 avgRadiance = radianceSum * invRayCount;
    if (closestIndices.x != GI_INVALID_INDEX)
    {
        gGiHitRecords.Store4(record + 0u,
            uint4(asuint(closestPosition), asuint(closestDistance)));
        gGiHitRecords.Store4(record + 16u,
            uint4(asuint(closestGeometricNormal), closestFlags));
        gGiHitRecords.Store4(record + 32u, uint4(asuint(closestShadingNormal),
            asuint(closestFrontFace ? 1.0f : 0.0f)));
        gGiHitRecords.Store4(record + 48u,
            uint4(asuint(closestUv), closestIndices.w, 0u));
        gGiHitRecords.Store4(record + 64u,
            uint4(asuint(avgRadiance), asuint(hitRatio)));
        gGiHitRecords.Store4(record + 80u, closestIndices);
        gGiHitRecords.Store4(record + 112u,
            uint4(closestGeneration, gGiTraceConstants.sceneGeneration,
                closestFlags, 1u));
    }
    else
    {
        gGiHitRecords.Store4(record + 0u, uint4(asuint(sourcePosition),
            asuint(giSceneRayMaxDistance())));
        gGiHitRecords.Store4(record + 16u, uint4(asuint(sourceNormal), 0u));
        gGiHitRecords.Store4(record + 32u,
            uint4(asuint(sourceNormal), asuint(0.0f)));
        gGiHitRecords.Store4(record + 48u, uint4(0u, 0u, GI_INVALID_INDEX, 0u));
        gGiHitRecords.Store4(record + 64u,
            uint4(asuint(avgRadiance), asuint(hitRatio)));
        gGiHitRecords.Store4(record + 80u, uint4(GI_INVALID_INDEX,
            GI_INVALID_INDEX, GI_INVALID_INDEX, GI_INVALID_INDEX));
        gGiHitRecords.Store4(record + 96u, uint4(0u, 0u, 0u, 0u));
        gGiHitRecords.Store4(record + 112u,
            uint4(0u, gGiTraceConstants.sceneGeneration, 0u, 2u));
    }
}

bool giTraceProbeRadiance(float3 origin, float3 direction, float maxDistance,
    uint seed, uint sampleIndex, out float3 radiance, out float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = giSceneRayMinDistance();
    ray.TMax = maxDistance;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(gGiSceneTlas, RAY_FLAG_NONE, 0xffu, ray);
    [loop] while (query.Proceed())
    {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) continue;
        const uint instanceId = query.CandidateInstanceID();
        if (instanceId >= gGiTraceConstants.instanceCount ||
            query.CandidateGeometryIndex() != 0u) continue;
        const GiInstance instance = giLoadInstance(instanceId);
        if (instance.identity.x != instanceId ||
            instance.identity.z >= gGiTraceConstants.geometryCount) continue;
        const GiGeometry geometry = giLoadGeometry(instance.identity.z);
        const uint primitiveIndex = query.CandidatePrimitiveIndex();
        if (geometry.mapping.x >= GI_MAX_MODEL_BUFFERS ||
            primitiveIndex * 3u + 2u >= geometry.range.w) continue;
        bool accepted = true;
        if ((geometry.mapping.z & GI_GEOMETRY_ALPHA_TESTED) != 0u)
        {
            const uint materialId = instance.state.z + geometry.mapping.y;
            if (materialId >= gGiTraceConstants.materialCount) accepted = false;
            else
            {
                const uint3 indices = giLoadTriangleIndices(geometry, primitiveIndex);
                const GiVertex v0 = giLoadVertex(geometry.mapping.x, indices.x);
                const GiVertex v1 = giLoadVertex(geometry.mapping.x, indices.y);
                const GiVertex v2 = giLoadVertex(geometry.mapping.x, indices.z);
                const float2 uv = giBarycentricUv(v0, v1, v2,
                    query.CandidateTriangleBarycentrics());
                const GiMaterial material = giLoadMaterial(materialId);
                float alpha = material.baseColor.a;
                if (giValidTextureSampler(material.baseColorTexture,
                    material.baseColorSampler))
                    alpha *= giSampleTexture(material.baseColorTexture,
                        material.baseColorSampler, uv).a;
                accepted = alpha >= material.alphaCutoff;
            }
        }
        if (accepted) query.CommitNonOpaqueTriangleHit();
        else gGiDiagnostics.InterlockedAdd(60u, 1u);
    }

    gGiDiagnostics.InterlockedAdd(48u, 1u);
    distance = maxDistance;
    radiance = 0.0f;
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        gGiDiagnostics.InterlockedAdd(56u, 1u);
        radiance = gGiTraceConstants.environmentEnabled != 0u
            ? max(gGiEnvironment.SampleLevel(gGiEnvironmentSampler,
                direction, 0.0f).rgb, 0.0f) *
                (gGiTraceConstants.environmentIntensity *
                    gGiFrame.environmentTransportExposure)
            : 0.0f;
        return false;
    }

    const uint instanceId = query.CommittedInstanceID();
    if (instanceId >= gGiTraceConstants.instanceCount ||
        query.CommittedGeometryIndex() != 0u) return false;
    const GiInstance instance = giLoadInstance(instanceId);
    const uint geometryIndex = instance.identity.z;
    if (instance.identity.x != instanceId ||
        geometryIndex >= gGiTraceConstants.geometryCount) return false;
    const GiGeometry geometry = giLoadGeometry(geometryIndex);
    const uint primitiveIndex = query.CommittedPrimitiveIndex();
    if (geometry.mapping.x >= GI_MAX_MODEL_BUFFERS ||
        primitiveIndex * 3u + 2u >= geometry.range.w) return false;
    const uint3 indices = giLoadTriangleIndices(geometry, primitiveIndex);
    const GiVertex v0 = giLoadVertex(geometry.mapping.x, indices.x);
    const GiVertex v1 = giLoadVertex(geometry.mapping.x, indices.y);
    const GiVertex v2 = giLoadVertex(geometry.mapping.x, indices.z);
    const float2 bary = query.CommittedTriangleBarycentrics();
    const float bw = 1.0f - bary.x - bary.y;
    const float3 worldPosition = giTransformPoint(instance,
        v0.position * bw + v1.position * bary.x + v2.position * bary.y);
    const float3 wp0 = giTransformPoint(instance, v0.position);
    const float3 wp1 = giTransformPoint(instance, v1.position);
    const float3 wp2 = giTransformPoint(instance, v2.position);
    float3 geometricNormal = normalize(cross(wp1 - wp0, wp2 - wp0));
    float3 shadingNormal = giTransformNormal(instance,
        normalize(v0.normal * bw + v1.normal * bary.x + v2.normal * bary.y));
    const float2 uv = giBarycentricUv(v0, v1, v2, bary);
    const uint materialId = instance.state.z + geometry.mapping.y;
    if (materialId >= gGiTraceConstants.materialCount) return false;
    const GiMaterial material = giLoadMaterial(materialId);
    const float4 tangent = v0.tangent * bw +
        v1.tangent * bary.x + v2.tangent * bary.y;
    if (giValidTextureSampler(material.normalTexture, material.normalSampler))
    {
        const float3 normalSample = giSampleTexture(material.normalTexture,
            material.normalSampler, uv).xyz * 2.0f - 1.0f;
        const float3 transformedTangent =
            giTransformDirection(instance, tangent.xyz);
        const float3 worldTangent = normalize(transformedTangent -
            shadingNormal * dot(transformedTangent, shadingNormal));
        const float3 worldBitangent = normalize(
            cross(shadingNormal, worldTangent) * tangent.w);
        shadingNormal = normalize(worldTangent * normalSample.x +
            worldBitangent * normalSample.y + shadingNormal * normalSample.z);
    }
    if (dot(geometricNormal, direction) > 0.0f) geometricNormal = -geometricNormal;
    if (dot(shadingNormal, direction) > 0.0f) shadingNormal = -shadingNormal;
    if (dot(shadingNormal, geometricNormal) < 0.0f) shadingNormal = -shadingNormal;

    float3 emissive = material.emissive.rgb;
    if (giValidTextureSampler(material.emissiveTexture, material.emissiveSampler))
        emissive *= giSampleTexture(material.emissiveTexture,
            material.emissiveSampler, uv).rgb;
    float3 albedo = material.baseColor.rgb;
    if (giValidTextureSampler(material.baseColorTexture, material.baseColorSampler))
        albedo *= giSampleTexture(material.baseColorTexture,
            material.baseColorSampler, uv).rgb;
    float metallic = saturate(material.metallicFactor);
    if (giValidTextureSampler(material.metallicRoughnessTexture,
        material.metallicRoughnessSampler))
        metallic *= giSampleTexture(material.metallicRoughnessTexture,
            material.metallicRoughnessSampler, uv).b;
    float ao = 1.0f;
    if (giValidTextureSampler(material.occlusionTexture, material.occlusionSampler))
        ao = lerp(1.0f, giSampleTexture(material.occlusionTexture,
            material.occlusionSampler, uv).r,
            saturate(material.occlusionStrength));
    const float3 directE = giDirectIrradiance(worldPosition,
        geometricNormal, shadingNormal, seed, sampleIndex);
    float feedbackConfidence = 0.0f;
    float3 feedbackE = 0.0f;
    if (gGiTraceConstants.feedbackEnabled != 0u)
    {
        feedbackE = giSampleProbeIrradiance(worldPosition,
            shadingNormal, feedbackConfidence);
        feedbackConfidence = min(feedbackConfidence, 0.75f);
        if (feedbackConfidence > 0.0f)
            gGiDiagnostics.InterlockedAdd(120u, 1u);
    }
    radiance = max(emissive, 0.0f) + max(albedo, 0.0f) * ao *
        (1.0f - saturate(metallic)) * (1.0f / GI_PI) *
        (directE + feedbackE * feedbackConfidence);
    distance = query.CommittedRayT();
    gGiDiagnostics.InterlockedAdd(52u, 1u);
    return true;
}

void giUpdateProbes(uint3 id)
{
    if (!giProbeFallbackEnabled()) return;
    if (gGiTraceConstants.freezeAfterFrame != 0u &&
        gGiTraceConstants.frameIndex >= gGiTraceConstants.freezeAfterFrame)
        return;
    const uint resolution = giProbeResolution();
    // A full cubic sweep spends most of a fixed game budget in empty sky. Keep
    // a camera-relative vertical slab per clip level while retaining the full
    // toroidal storage so newly revealed floors/rooms stream without allocation.
    const uint verticalSlices = min(resolution, 8u);
    const uint scheduledPerLevel = resolution * resolution * verticalSlices;
    const uint totalProbes = scheduledPerLevel * giClipmapCount();
    const uint updateBudget = max(
        (uint)round(gGiTraceConstants.confidenceBlend), 1u);
    if (id.x >= updateBudget || totalProbes == 0u) return;
    const uint ordinal =
        (giProbeRelativeFrame() % giProbeSweepLength()) * updateBudget + id.x;
    if (ordinal >= totalProbes) return;
    const uint level = ordinal / scheduledPerLevel;
    uint localOrdinal = ordinal - level * scheduledPerLevel;
    const uint lx = localOrdinal % resolution;
    localOrdinal /= resolution;
    const uint lz = localOrdinal % resolution;
    const uint ly = localOrdinal / resolution;
    const float spacing = giProbeSpacing(level);
    const int3 cameraCell = int3(floor(gGiFrame.cameraPosition / spacing));
    const int halfResolution = int(resolution / 2u);
    const int halfVertical = int(verticalSlices / 2u);
    const int3 targetCell = cameraCell + int3(int(lx) - halfResolution,
        int(ly) - halfVertical, int(lz) - halfResolution);
    const float3 targetPosition = (float3(targetCell) + 0.5f) * spacing;
    const uint probeKey = giCellKeyAtLevel(targetCell, level);
    const uint address = giProbeIndex(targetCell, level) * GI_PROBE_STRIDE;
    const bool sourceIsStaging = giProbeSourceIsStaging();
    const uint4 oldFlags = giLoadProbe4(sourceIsStaging, address + 112u);
    const bool oldResident = (oldFlags.w & 2u) != 0u;
    const bool oldClassified = (oldFlags.w & 1u) != 0u;
    const bool sameWorldCell = oldResident && oldFlags.x == probeKey &&
        oldFlags.y == level;
    const float4 oldPositionConfidence = asfloat(
        giLoadProbe4(sourceIsStaging, address + 96u));
    const float3 traceOrigin = sameWorldCell
        ? oldPositionConfidence.xyz : targetPosition;
    // Probe-primary tracing consumes the explicit fixed ray budget directly.
    // Thirty-two stable directions materially reduce low-order SH and depth
    // moment error without relying on temporal noise to converge.
    const uint rayCount = clamp(max(
        gGiTraceConstants.rayBudget / updateBudget, 1u), 4u, 32u);
    const float maxDistance = min(giSceneRayMaxDistance(), spacing * 8.0f);
    float3 c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float distanceSum = 0.0f, distanceSquareSum = 0.0f;
    float lumaSum = 0.0f, lumaSquareSum = 0.0f;
    float hitCount = 0.0f;
    float3 nearDirection = 0.0f;
    float4 axisSum0 = 0.0f, axisWeight0 = 0.0f;
    float2 axisSum1 = 0.0f, axisWeight1 = 0.0f;
    [loop] for (uint rayIndex = 0u; rayIndex < rayCount; ++rayIndex)
    {
        const float3 direction = giProbeSphereDirection(
            probeKey, rayIndex, rayCount);
        float3 rayRadiance;
        float rayDistance;
        const bool hit = giTraceProbeRadiance(traceOrigin, direction,
            maxDistance, probeKey, rayIndex, rayRadiance, rayDistance);
        const float storedDistance = min(rayDistance, maxDistance);
        const float4 basis = giShBasis(direction);
        c0 += rayRadiance * basis.x;
        c1 += rayRadiance * basis.y;
        c2 += rayRadiance * basis.z;
        c3 += rayRadiance * basis.w;
        distanceSum += storedDistance;
        distanceSquareSum += storedDistance * storedDistance;
        const float luma = dot(rayRadiance, float3(0.2126f, 0.7152f, 0.0722f));
        lumaSum += luma;
        lumaSquareSum += luma * luma;
        if (hit)
        {
            hitCount += 1.0f;
            const float nearWeight = saturate(1.0f -
                storedDistance / max(spacing * 0.35f, 1.0e-3f));
            nearDirection += direction * nearWeight;
        }
        const float4 w0 = float4(max(direction.x, 0.0f),
            max(-direction.x, 0.0f), max(direction.y, 0.0f),
            max(-direction.y, 0.0f));
        const float2 w1 = float2(max(direction.z, 0.0f),
            max(-direction.z, 0.0f));
        axisSum0 += storedDistance * w0;
        axisWeight0 += w0;
        axisSum1 += storedDistance * w1;
        axisWeight1 += w1;
    }
    const float invRayCount = rcp(float(rayCount));
    const float shScale = 12.56637061436f * invRayCount;
    const float4 frameR = float4(c0.x, c1.x, c2.x, c3.x) * shScale;
    const float4 frameG = float4(c0.y, c1.y, c2.y, c3.y) * shScale;
    const float4 frameB = float4(c0.z, c1.z, c2.z, c3.z) * shScale;
    const float meanDistance = distanceSum * invRayCount;
    const float depthVariance = max(distanceSquareSum * invRayCount -
        meanDistance * meanDistance, spacing * spacing * 1.0e-4f);
    const float meanLuma = lumaSum * invRayCount;
    const float radianceVariance = max(lumaSquareSum * invRayCount -
        meanLuma * meanLuma, 0.0f);
    const bool partiallyOpen = hitCount > 0.0f &&
        hitCount < float(rayCount);
    const float classificationThreshold = oldClassified && sameWorldCell
        ? 0.08f : 0.14f;
    const bool freshClassified = hitCount > 0.0f && (partiallyOpen ||
        meanDistance > spacing * classificationThreshold);
    // Classification is a geometric property (does this world cell contain an
    // occluder?), and the geometry of a resident cell does not change frame to
    // frame. Recomputing it every update let the binary flag toggle on tiny
    // trace variation for probes sitting near the threshold, flipping their
    // gather weight between a value and zero -> visible dark-area flicker on
    // walls/doorways. Latch it once the cell is resident; a genuine geometry
    // change re-assigns the cell (sameWorldCell == false) and reclassifies.
    const bool classified = sameWorldCell ? oldClassified : freshClassified;
    float3 relocatedPosition = targetPosition;
    if (dot(nearDirection, nearDirection) > 1.0e-5f)
        relocatedPosition -= normalize(nearDirection) * min(spacing * 0.35f,
            length(nearDirection) * spacing * 0.08f);
    // Relocation is geometric and the geometry of a resident cell is static, so
    // the probe must settle once and then STOP. The discrete set of near-hits made
    // the relocation target jump as the probe moved, producing a limit cycle that
    // the old tiny deadband could not damp: the position micro-oscillated every
    // update, moving the trace origin and flickering the cached/gathered/GI-only
    // debug views. The relocation target is always within 0.35*spacing of the cell
    // centre (the offset is capped), so the fresh-frame placement already reaches
    // it; latch the position for every resident update afterwards.
    if (sameWorldCell)
    {
        relocatedPosition = oldPositionConfidence.xyz;
    }
    const float sampleConfidence = classified ? saturate(float(rayCount) / 16.0f) /
        (1.0f + sqrt(radianceVariance)) : 0.0f;
    // Probe rays and lighting inputs are deterministic for a resident cell.  A
    // fractional blend therefore does not suppress noise; it turns each probe's
    // scheduled update into a long 18-frame-step convergence sequence.  The
    // destination field is now preserved from the immutable completed source
    // every frame, so publish the complete deterministic estimate directly.
    const float blend = 1.0f;
    const float4 oldR = asfloat(giLoadProbe4(sourceIsStaging, address + 0u));
    const float4 oldG = asfloat(giLoadProbe4(sourceIsStaging, address + 16u));
    const float4 oldB = asfloat(giLoadProbe4(sourceIsStaging, address + 32u));
    const float4 oldDepth = asfloat(
        giLoadProbe4(sourceIsStaging, address + 48u));
    const float4 oldAxis0 = asfloat(
        giLoadProbe4(sourceIsStaging, address + 64u));
    const float4 oldAxis1 = asfloat(
        giLoadProbe4(sourceIsStaging, address + 80u));
    const float4 axisDepth0 = axisSum0 / max(axisWeight0, 1.0e-4f.xxxx);
    const float2 axisDepth1 = axisSum1 / max(axisWeight1, 1.0e-4f.xx);
    const float4 frameDepth = float4(meanDistance, depthVariance,
        hitCount * invRayCount, radianceVariance);
    const float4 frameAxis1 = float4(axisDepth1, 0.0f, 0.0f);
    giStoreProbeDestination4(sourceIsStaging, address + 0u,
        asuint(lerp(oldR, frameR, blend)));
    giStoreProbeDestination4(sourceIsStaging, address + 16u,
        asuint(lerp(oldG, frameG, blend)));
    giStoreProbeDestination4(sourceIsStaging, address + 32u,
        asuint(lerp(oldB, frameB, blend)));
    giStoreProbeDestination4(sourceIsStaging, address + 48u,
        asuint(lerp(oldDepth, frameDepth, blend)));
    giStoreProbeDestination4(sourceIsStaging, address + 64u,
        asuint(lerp(oldAxis0, axisDepth0, blend)));
    giStoreProbeDestination4(sourceIsStaging, address + 80u,
        asuint(lerp(oldAxis1, frameAxis1, blend)));
    const float confidence = sameWorldCell
        ? lerp(oldPositionConfidence.w, sampleConfidence, blend)
        : sampleConfidence;
    giStoreProbeDestination4(sourceIsStaging, address + 96u,
        uint4(asuint(relocatedPosition), asuint(confidence)));
    // Keep residency separate from classification. Previously an unclassified
    // frame erased the world-cell identity, so borderline probes repeatedly
    // reinitialized and visibly popped near doors and interior walls.
    giStoreProbeDestination4(sourceIsStaging, address + 112u,
        uint4(probeKey, level,
        gGiTraceConstants.frameIndex, 2u | (classified ? 1u : 0u)));
    if (!oldClassified && classified)
        gGiResidualInterface.InterlockedAdd(4u, 1u);
    else if (oldClassified && !classified)
        giAtomicSaturatingDecrement(gGiResidualInterface, 4u);
    gGiDiagnostics.InterlockedAdd(116u, 1u);
    gGiDiagnostics.InterlockedAdd(28u, rayCount);
}

float3 giDebugColor(uint2 pixel, uint width)
{
    const uint workCount = max(gGiCounters.Load(0u), 1u);
    const uint index = (pixel.y * width + pixel.x) % workCount;
    const uint record = index * GI_HIT_RECORD_STRIDE;
    if (gGiTraceConstants.debugView == 6u)
    {
        const float hitRatio = asfloat(gGiHitRecords.Load(record + 76u));
        return lerp(float3(0.05f, 0.1f, 0.4f), float3(1.0f, 0.25f, 0.05f), hitRatio);
    }
    if (gGiTraceConstants.debugView == 7u)
    {
        const float distance = asfloat(gGiHitRecords.Load(record + 12u));
        return (1.0f - exp(-distance * 0.05f)).xxx;
    }
    if (gGiTraceConstants.debugView == 8u)
        return asfloat(gGiHitRecords.Load3(record + 32u)) * 0.5f + 0.5f;
    if (gGiTraceConstants.debugView == 9u)
    {
        const uint material = gGiHitRecords.Load(record + 56u);
        return float3(frac(float(material) * 0.6180339f),
            frac(float(material) * 0.381966f), frac(float(material) * 0.173205f));
    }
    return 0.0f;
}

// Reconstruct diffuse irradiance E(n) from an L1 radiance SH via the clamped
// cosine-lobe convolution (A0 = pi, A1 = 2pi/3).
float3 giIrradianceFromSH(float4 shR, float4 shG, float4 shB, float3 n)
{
    const float4 wY = float4(GI_PI, 2.09439510f, 2.09439510f, 2.09439510f) *
        giShBasis(n);
    return max(float3(dot(shR, wY), dot(shG, wY), dot(shB, wY)), 0.0f);
}

void giEvaluate(uint3 id)
{
    uint width = 0, height = 0;
    gGiOutput.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) return;
    const uint sIndex = id.y * width + id.x;
    const GiSurface surf = giLoadSurface(sIndex);

    // Debug visualizations bypass the shading output.
    if (gGiTraceConstants.debugView == 8u)
    {
        gGiOutput[id.xy] = float4(surf.valid ? surf.normal * 0.5f + 0.5f : 0.0f, 1.0f);
        return;
    }
    if (gGiTraceConstants.debugView == 9u)
    {
        const float material = surf.valid ? float(surf.materialId) : 0.0f;
        gGiOutput[id.xy] = float4(
            frac(material * float3(0.6180339f, 0.381966f, 0.173205f)), 1.0f);
        return;
    }

    float3 irradiance = 0.0f;
    float coverage = 0.0f;
    float directVisibility = 1.0f;
    if (surf.valid)
    {
        const float3 sunDirection = normalize(-gGiFrame.lightDirection);
        if (dot(surf.normal, sunDirection) > 0.0f)
        {
            const float offset = giOriginOffset(
                surf.position, surf.normal, 1.0e-4f);
            directVisibility = giVisible(surf.position + surf.normal * offset,
                sunDirection, giSceneRayMaxDistance()) ? 1.0f : 0.0f;
        }
        uint selectedSurfel = GI_INVALID_INDEX;
        float selectedWeight = 0.0f;
        if (giSurfelDetailEnabled())
        {
        const uint clipLevel = giSurfelClipLevel(surf.position);
        const float levelCellSize = giLevelCellSize(clipLevel);
        const int3 baseCell = giCellCoordAtLevel(
            surf.position, clipLevel);
        const uint bucketCount = max(gGiTraceConstants.hashBucketCount, 1u);
        const uint cands = clamp(gGiTraceConstants.candidatesPerCell, 1u, 4u);
        const uint maxSurfels = max(gGiTraceConstants.maxSurfels, 1u);
        const float gatherRadius = max(gGiTraceConstants.gatherRadiusScale *
            levelCellSize, 1.0e-4f);
        const float normalDenom =
            max(1.0f - gGiTraceConstants.normalThreshold, 1.0e-3f);
        float3 sum = 0.0f;
        float weight = 0.0f;
        uint gathered = 0u;

        [loop] for (int oz = -1; oz <= 1; ++oz)
        [loop] for (int oy = -1; oy <= 1; ++oy)
        [loop] for (int ox = -1; ox <= 1; ++ox)
        {
            const uint key = giCellKeyAtLevel(
                baseCell + int3(ox, oy, oz), clipLevel);
            const uint4 slots = gGiHashBuckets.Load4((key % bucketCount) * 16u);
            [loop] for (uint k = 0u; k < cands; ++k)
            {
                const uint si = slots[k];
                if (si == GI_INVALID_INDEX || si >= maxSurfels) continue;
                const GiSurfelData s = giLoadSurfel(si);
                if (s.cellKey != key ||
                    s.sceneGeneration != gGiTraceConstants.sceneGeneration ||
                    s.confidence <= 0.0f)
                    continue;
                if ((s.flags & 7u) != clipLevel) continue;
                if (s.materialId != surf.materialId ||
                    s.instanceId != surf.instanceId)
                    continue;
                const float3 delta = s.position - surf.position;
                const float dist = length(delta);
                const float adaptiveGatherRadius = max(
                    gatherRadius, s.radius * 2.0f);
                if (dist > adaptiveGatherRadius) continue;
                const float nAlign = dot(normalize(s.normal), surf.normal);
                if (nAlign < gGiTraceConstants.normalThreshold) continue;
                // Plane rejection blocks leaks across thin walls / adjacent rooms.
                const float gatherPlaneThreshold = max(
                    gGiTraceConstants.planeThreshold,
                    min(levelCellSize * 0.15f, 0.35f));
                if (abs(dot(delta, surf.normal)) > gatherPlaneThreshold)
                    continue;
                const float wDist = saturate(
                    1.0f - dist / adaptiveGatherRadius);
                const float wNorm = saturate(
                    (nAlign - gGiTraceConstants.normalThreshold) / normalDenom);
                const float w = wDist * wNorm * s.confidence;
                if (w <= 0.0f) continue;
                const uint a = si * GI_SURFEL_STRIDE;
                const float3 shIrradiance = giIrradianceFromSH(
                    asfloat(gGiSurfels.Load4(a + 80u)),
                    asfloat(gGiSurfels.Load4(a + 96u)),
                    asfloat(gGiSurfels.Load4(a + 112u)), surf.normal);
                const float3 exactNormalIrradiance = max(asfloat(
                    gGiVisibilityMoments.Load3(si * 32u + 16u)), 0.0f);
                const float exactWeight = saturate(
                    (nAlign - 0.95f) / 0.05f);
                sum += w * lerp(shIrradiance,
                    exactNormalIrradiance, exactWeight);
                weight += w;
                ++gathered;
                if (w > selectedWeight)
                {
                    selectedWeight = w;
                    selectedSurfel = si;
                }
            }
        }

        if (weight > 1.0e-4f)
        {
            irradiance = sum / weight;
            coverage = saturate(weight);
        }
        }
        float probeConfidence = 0.0f;
        GiProbeLevelSample probeDebug;
        const float3 probeIrradiance = giSampleProbeIrradianceDetailed(
            surf.position, surf.normal, probeConfidence, probeDebug);
        if ((gGiTraceConstants.giFlags & 1u) != 0u)
        {
            const int traceSlot = giTracePixelSlot(id.xy, uint2(width, height));
            if (traceSlot >= 0)
            {
                const uint traceAddress = giTracePixelAddress(traceSlot);
                const bool sourceIsStaging = giProbeReadIsStaging();
                gGiDiagnostics.Store4(traceAddress + 0u, uint4(
                    probeDebug.selectedIndex, sourceIsStaging ? 1u : 0u,
                    giProbeRelativeFrame() % giProbeSweepLength(), 0u));
                float4 traceR = 0.0f, traceG = 0.0f;
                float4 traceB = 0.0f, traceDepth = 0.0f;
                if (probeDebug.selectedIndex != GI_INVALID_INDEX)
                {
                    const uint probeAddress =
                        probeDebug.selectedIndex * GI_PROBE_STRIDE;
                    traceR = asfloat(giLoadProbe4(
                        sourceIsStaging, probeAddress + 0u));
                    traceG = asfloat(giLoadProbe4(
                        sourceIsStaging, probeAddress + 16u));
                    traceB = asfloat(giLoadProbe4(
                        sourceIsStaging, probeAddress + 32u));
                    traceDepth = asfloat(giLoadProbe4(
                        sourceIsStaging, probeAddress + 48u));
                }
                gGiDiagnostics.Store4(traceAddress + 16u, asuint(traceR));
                gGiDiagnostics.Store4(traceAddress + 32u, asuint(traceG));
                gGiDiagnostics.Store4(traceAddress + 48u, asuint(traceB));
                gGiDiagnostics.Store4(traceAddress + 64u, asuint(traceDepth));
                gGiDiagnostics.Store4(traceAddress + 80u, asuint(float4(
                    probeIrradiance, probeConfidence)));
                gGiDiagnostics.Store4(traceAddress + 144u, asuint(float4(
                    directVisibility, 0.0f, 0.0f, 0.0f)));
                // Record every fine-level trilinear contributor.  The dominant
                // entry alone can remain stable while another entry is evicted
                // or rewritten, so the bounded trace needs the complete gather
                // footprint to identify the exact physical probe transition.
                const float traceNearExtent = max(
                    gGiTraceConstants.cellSize *
                    float(giProbeResolution()) * 1.6f, 1.0e-3f);
                const float traceContinuousLevel = clamp(log2(max(length(
                    surf.position - gGiFrame.cameraPosition) /
                    traceNearExtent, 1.0f)), 0.0f,
                    float(giClipmapCount() - 1u));
                const uint traceLevel = min(
                    (uint)floor(traceContinuousLevel), giClipmapCount() - 1u);
                const float traceSpacing = giProbeSpacing(traceLevel);
                const int3 traceBaseCell = int3(floor(
                    surf.position / traceSpacing - 0.5f));
                [unroll] for (uint traceProbe = 0u;
                    traceProbe < 8u; ++traceProbe)
                {
                    const int3 cell = traceBaseCell + int3(
                        int(traceProbe & 1u),
                        int((traceProbe >> 1u) & 1u),
                        int((traceProbe >> 2u) & 1u));
                    const uint index = giProbeIndex(cell, traceLevel);
                    const uint probeAddress = index * GI_PROBE_STRIDE;
                    const uint4 flags = giLoadProbe4(
                        sourceIsStaging, probeAddress + 112u);
                    const uint outputAddress = traceAddress +
                        GI_TRACE_PROBE_BASE +
                        traceProbe * GI_TRACE_PROBE_STRIDE;
                    gGiDiagnostics.Store4(outputAddress + 0u, uint4(
                        index, giCellKeyAtLevel(cell, traceLevel), flags.x,
                        (flags.y & 0xffu) | (flags.w << 8u)));
                    gGiDiagnostics.Store4(outputAddress + 16u,
                        giLoadProbe4(sourceIsStaging, probeAddress + 0u));
                    gGiDiagnostics.Store4(outputAddress + 32u,
                        giLoadProbe4(sourceIsStaging, probeAddress + 16u));
                    gGiDiagnostics.Store4(outputAddress + 48u,
                        giLoadProbe4(sourceIsStaging, probeAddress + 32u));
                    gGiDiagnostics.Store4(outputAddress + 64u,
                        giLoadProbe4(sourceIsStaging, probeAddress + 48u));
                }
            }
        }
        if (gGiTraceConstants.debugView == 6u ||
             gGiTraceConstants.debugView == 7u ||
             gGiTraceConstants.debugView == 10u ||
             gGiTraceConstants.debugView == 11u ||
             gGiTraceConstants.debugView == 12u ||
             gGiTraceConstants.debugView == 13u ||
             gGiTraceConstants.debugView == 16u ||
             gGiTraceConstants.debugView == 17u ||
             gGiTraceConstants.debugView == 19u ||
             gGiTraceConstants.debugView == 20u)
        {
            float3 debug = 0.0f;
            uint4 selectedFlags = 0u;
            float4 selectedDepth = 0.0f;
            if (probeDebug.selectedIndex != GI_INVALID_INDEX)
            {
                const bool completedIsStaging = giProbeReadIsStaging();
                const uint selectedAddress =
                    probeDebug.selectedIndex * GI_PROBE_STRIDE;
                selectedFlags = giLoadProbe4(
                    completedIsStaging, selectedAddress + 112u);
                selectedDepth = asfloat(giLoadProbe4(
                    completedIsStaging, selectedAddress + 48u));
            }
            const bool selectedValid =
                probeDebug.selectedIndex != GI_INVALID_INDEX;
            const float selectedLevel = float(selectedFlags.y);
            const float selectedMaxDistance = max(
                giProbeSpacing(selectedFlags.y) * 8.0f, 1.0e-4f);
            const float hitRatio = saturate(selectedDepth.z);
            const float updated = selectedValid &&
                selectedFlags.z == gGiTraceConstants.frameIndex ? 1.0f : 0.0f;
            if (gGiTraceConstants.debugView == 6u)
                debug = selectedValid
                    ? float3(0.0f, hitRatio, 1.0f - hitRatio) : 0.0f;
            else if (gGiTraceConstants.debugView == 7u)
                debug = selectedValid
                    ? saturate(selectedDepth.x / selectedMaxDistance).xxx : 0.0f;
            else if (gGiTraceConstants.debugView == 10u && selectedValid)
            {
                const float3 identity = float3(
                    frac(float(probeDebug.selectedIndex) * 0.6180339f),
                    frac(float(probeDebug.selectedIndex) * 0.381966f),
                    frac(float(probeDebug.selectedIndex) * 0.173205f));
                const float levelScale = lerp(1.0f, 0.35f,
                    selectedLevel / max(float(giClipmapCount() - 1u), 1.0f));
                debug = identity * levelScale;
            }
            else if (gGiTraceConstants.debugView == 11u)
                debug = float3(saturate(probeDebug.relocation / 0.35f),
                    probeConfidence, (selectedFlags.w & 1u) != 0u ? 1.0f : 0.0f);
            else if (gGiTraceConstants.debugView == 12u)
                debug = float3(saturate(selectedDepth.x / selectedMaxDistance),
                    saturate(sqrt(max(selectedDepth.y, 0.0f)) /
                        selectedMaxDistance), probeDebug.visibility);
            else if (gGiTraceConstants.debugView == 13u)
                debug = probeDebug.rawRadiance;
            else if (gGiTraceConstants.debugView == 16u)
                debug = float3(saturate(probeDebug.age / 512.0f), updated,
                    (selectedFlags.w & 2u) != 0u ? 1.0f : 0.0f);
            else if (gGiTraceConstants.debugView == 17u)
                debug = float3(saturate(probeDebug.variance * 8.0f),
                    probeConfidence, probeDebug.visibility);
            else if (gGiTraceConstants.debugView == 19u)
                debug = selectedValid
                    ? float3(updated, hitRatio, 1.0f - hitRatio) : 0.0f;
            else if (gGiTraceConstants.debugView == 20u)
                debug = probeIrradiance * (1.0f / GI_PI);
            gGiOutput[id.xy] = float4(debug,
                giPackCoverageVisibility(probeConfidence, directVisibility));
            return;
        }
        if (!giSurfelDetailEnabled())
        {
            irradiance = probeIrradiance;
            coverage = probeConfidence;
        }
        else if (probeConfidence > 0.0f && coverage < 0.95f)
        {
            const float surfelBlend = coverage > 0.0f
                ? saturate(coverage / (coverage + probeConfidence))
                : 0.0f;
            irradiance = lerp(probeIrradiance, irradiance, surfelBlend);
            coverage = saturate(coverage +
                probeConfidence * (1.0f - coverage));
            gGiDiagnostics.InterlockedAdd(124u, 1u);
        }
        if (coverage <= 0.0f)
        {
            gGiDiagnostics.InterlockedAdd(24u, 1u); // rejectedGathers (uncovered)
        }

    }

    // Bounded fixed-point reductions keep diagnostics deterministic and avoid
    // relying on optional floating-point atomics. Offset 16 is the irradiance
    // luminance sum (Q8, clamped to 16 per pixel), 20 is the positive-float
    // maximum, 80 is valid gather count, and 84 is coverage sum (Q12).
    if ((gGiTraceConstants.giFlags & 1u) != 0u && coverage > 0.0f)
    {
        const float irradianceLuma = dot(max(irradiance, 0.0f),
            float3(0.2126f, 0.7152f, 0.0722f));
        gGiDiagnostics.InterlockedAdd(80u, 1u);
        gGiDiagnostics.InterlockedAdd(16u,
            (uint)round(min(irradianceLuma, 16.0f) * 256.0f));
        gGiDiagnostics.InterlockedMax(20u, asuint(irradianceLuma));
        gGiDiagnostics.InterlockedAdd(84u,
            (uint)round(coverage * 4096.0f));
        gGiDiagnostics.InterlockedAdd(96u,
            (uint)round(min(irradiance.r, 16.0f) * 256.0f));
        gGiDiagnostics.InterlockedAdd(100u,
            (uint)round(min(irradiance.g, 16.0f) * 256.0f));
        gGiDiagnostics.InterlockedAdd(104u,
            (uint)round(min(irradiance.b, 16.0f) * 256.0f));
    }

    // Store raw diffuse irradiance divided by pi so the clustered-forward
    // injection (albedo * value) evaluates the Lambert BRDF correctly.
    gGiOutput[id.xy] = float4(irradiance * (1.0f / GI_PI),
        giPackCoverageVisibility(coverage, directVisibility));
}

void giTemporal(uint3 id)
{
    uint width = 0, height = 0;
    gGiOutput.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) return;

    uint rawWidth = 0, rawHeight = 0;
    gGiPrimaryInput.GetDimensions(rawWidth, rawHeight);
    const uint2 pixel = id.xy;
    const float depth = gGiSceneDepth.Load(int3(pixel, 0));
    const bool reversedZ = gGiFrame.renderExtentAndHiZ.w != 0u;
    const bool background = reversedZ ? depth <= 0.0f : depth >= 1.0f;
    if (background)
    {
        gGiOutput[pixel] = 0.0f;
        return;
    }

    const float2 uv = (float2(pixel) + 0.5f) / float2(width, height);
    const float3 worldPosition = giWorldFromDepth(uv, depth);
    const float4 attributes = gGiSurfaceAttributes.Load(int3(pixel, 0));
    const float3 worldNormal = giDecodeOctahedralNormal(attributes.xy);
    const uint materialId = (uint)round(max(attributes.z, 0.0f));
    const uint instanceId = (uint)round(max(attributes.w, 0.0f));
    const uint packedIdentity = (instanceId << 16u) | materialId;

    // Edge-aware reconstruction from the reduced cache evaluation. Every tap
    // must represent the same instance/material and lie on the same plane.
    const uint divisor = max(gGiTraceConstants.evaluationDivisor, 1u);
    const int2 rawCenter = min(int2(pixel / divisor),
        int2(int(rawWidth) - 1, int(rawHeight) - 1));
    float4 weighted = 0.0f;
    float totalWeight = 0.0f;
    float3 mean = 0.0f;
    float3 meanSquare = 0.0f;
    float directVisibilitySum = 0.0f;
    float directVisibilityWeight = 0.0f;
    uint accepted = 0u;
    [unroll] for (int oy = -1; oy <= 1; ++oy)
    [unroll] for (int ox = -1; ox <= 1; ++ox)
    {
        const int2 rp = clamp(rawCenter + int2(ox, oy), 0,
            int2(int(rawWidth) - 1, int(rawHeight) - 1));
        const GiSurface sampleSurface = giLoadSurface(
            uint(rp.y) * rawWidth + uint(rp.x));
        if (!sampleSurface.valid ||
            sampleSurface.materialId != materialId ||
            sampleSurface.instanceId != instanceId)
            continue;
        const float normalAlignment = dot(sampleSurface.normal, worldNormal);
        if (normalAlignment < 0.85f) continue;
        const float3 delta = sampleSurface.position - worldPosition;
        if (abs(dot(delta, worldNormal)) >
                max(gGiTraceConstants.planeThreshold, 0.01f))
            continue;
        const float4 raw = gGiPrimaryInput.Load(int3(rp, 0));
        float rawCoverage = 0.0f;
        float rawDirectVisibility = 1.0f;
        giUnpackCoverageVisibility(raw.a, rawCoverage,
            rawDirectVisibility);
        if (!all(isfinite(raw))) continue;
        const float spatial = exp(-0.5f * float(ox * ox + oy * oy));
        const float baseWeight = spatial * normalAlignment * normalAlignment;
        directVisibilitySum += rawDirectVisibility * baseWeight;
        directVisibilityWeight += baseWeight;
        if (rawCoverage <= 0.0f) continue;
        const float w = baseWeight * rawCoverage;
        weighted += float4(raw.rgb, rawCoverage) * w;
        totalWeight += w;
        mean += raw.rgb;
        meanSquare += raw.rgb * raw.rgb;
        ++accepted;
    }

    float4 current = totalWeight > 1.0e-5f
        ? weighted / totalWeight : 0.0f;
    current.a = saturate(current.a);
    const float4 temporalCurrentInput = current;
    float currentDirectVisibility = directVisibilityWeight > 1.0e-5f
        ? saturate(directVisibilitySum / directVisibilityWeight) : 1.0f;
    const float invAccepted = accepted > 0u ? rcp(float(accepted)) : 0.0f;
    mean *= invAccepted;
    const float3 variance = max(meanSquare * invAccepted - mean * mean, 0.0f);
    const float3 sigma = sqrt(variance + 1.0e-6f);

    if ((gGiTraceConstants.debugView >= 6u &&
         gGiTraceConstants.debugView <= 9u) ||
        gGiTraceConstants.debugView == 18u ||
        gGiTraceConstants.debugView == 21u ||
        (gGiTraceConstants.debugView >= 10u &&
         gGiTraceConstants.debugView <= 13u) ||
        gGiTraceConstants.debugView == 16u ||
        gGiTraceConstants.debugView == 17u ||
        gGiTraceConstants.debugView == 19u ||
        gGiTraceConstants.debugView == 20u)
    {
        gGiOutput[pixel] = gGiPrimaryInput.Load(int3(rawCenter, 0));
        return;
    }

    uint rejectionReason = 0u;
    float4 history = 0.0f;
    float historyDirectVisibility = 1.0f;
    // Fraction of the reprojected bilinear footprint that landed on the same
    // world surface. Kept continuous so partial validity at edges scales the
    // history weight smoothly instead of a binary accept/reject that flickers.
    float historyConfidence = 0.0f;
    // History validity depends on the reprojected surface, not on whether this
    // frame happened to find probe support. Coupling the two dropped stable
    // history to black for one-frame coverage holes.
    bool historyValid = false;
    const float4 previousClip = mul(gGiFrame.previousViewProjection,
        float4(worldPosition, 1.0f));
    if (previousClip.w <= 1.0e-6f)
    {
        rejectionReason = 1u;
    }
    else
    {
        const float3 previousNdc = previousClip.xyz / previousClip.w;
#if defined(IC_TARGET_VULKAN)
        const float2 previousUv = previousNdc.xy * 0.5f + 0.5f;
#else
        const float2 previousUv = float2(previousNdc.x * 0.5f + 0.5f,
            0.5f - previousNdc.y * 0.5f);
#endif
        if (any(previousUv <= 0.0f) || any(previousUv >= 1.0f))
        {
            rejectionReason = 2u;
        }
        else
        {
            // Validated bilinear reprojection. Point-sampling the previous frame
            // at a truncated pixel snapped the tap onto neighbouring surfaces as
            // the camera moved, producing per-frame identity/normal rejections
            // that read as crawling, wavy indirect shadows. Each of the four
            // taps reconstructs its own world position (from its own pixel and
            // depth) and is accepted independently, so sub-pixel motion resolves
            // smoothly and only genuinely disoccluded footprints drop history.
            const float4x4 previousInverse = giInverse(
                gGiFrame.previousViewProjection);
            // Tolerance scales with view distance so the bilinear footprint of
            // an oblique surface (adjacent texels span more world space when
            // grazing) is not spuriously position-rejected. Identity and normal
            // tests still guard against cross-surface ghosting; this only gates
            // same-surface disocclusion, which produces far larger jumps.
            const float viewDistance =
                length(worldPosition - gGiFrame.cameraPosition);
            const float grazing = saturate(1.0f -
                abs(dot(worldNormal, normalize(
                    gGiFrame.cameraPosition - worldPosition))));
            const float positionTolerance = max(0.05f,
                0.01f * viewDistance) * (1.0f + 3.0f * grazing);
            const float2 sampleCoord =
                previousUv * float2(width, height) - 0.5f;
            const float2 baseCoordF = floor(sampleCoord);
            const float2 fracCoord = sampleCoord - baseCoordF;
            const int2 baseCoord = int2(baseCoordF);
            float3 historyColorSum = 0.0f;
            float historyCoverageSum = 0.0f;
            float historyVisSum = 0.0f;
            float validWeight = 0.0f;
            uint tapReject = 5u;
            [unroll] for (int ty = 0; ty < 2; ++ty)
            [unroll] for (int tx = 0; tx < 2; ++tx)
            {
                const float wX = tx == 0 ? 1.0f - fracCoord.x : fracCoord.x;
                const float wY = ty == 0 ? 1.0f - fracCoord.y : fracCoord.y;
                const float tapWeight = wX * wY;
                if (tapWeight <= 0.0f) continue;
                const int2 tapCoord = clamp(baseCoord + int2(tx, ty),
                    int2(0, 0), int2(int(width) - 1, int(height) - 1));
                const float tapDepth =
                    gGiPreviousSceneDepth.Load(int3(tapCoord, 0));
                const float4 tapAttr =
                    gGiPreviousSurfaceAttributes.Load(int3(tapCoord, 0));
                const uint tapIdentity =
                    ((uint)round(max(tapAttr.w, 0.0f)) << 16u) |
                    (uint)round(max(tapAttr.z, 0.0f));
                const float3 tapNormal = giDecodeOctahedralNormal(tapAttr.xy);
                const float2 tapUv =
                    (float2(tapCoord) + 0.5f) / float2(width, height);
                float2 tapNdc = tapUv * 2.0f - 1.0f;
#if !defined(IC_TARGET_VULKAN)
                tapNdc.y = -tapNdc.y;
#endif
                const float4 tapWorldH = mul(previousInverse,
                    float4(tapNdc, tapDepth, 1.0f));
                const float3 tapWorld = tapWorldH.xyz /
                    max(abs(tapWorldH.w), 1.0e-6f);
                if (tapIdentity != packedIdentity)
                {
                    tapReject = 3u; continue;
                }
                if (dot(tapNormal, worldNormal) < 0.9f)
                {
                    tapReject = 4u; continue;
                }
                if (distance(tapWorld, worldPosition) > positionTolerance)
                {
                    tapReject = 5u; continue;
                }
                const float4 tapHistory =
                    gGiHistoryInput.Load(int3(tapCoord, 0));
                if (!all(isfinite(tapHistory)))
                {
                    tapReject = 6u; continue;
                }
                float tapCoverage = 0.0f, tapVisibility = 1.0f;
                giUnpackCoverageVisibility(tapHistory.a, tapCoverage,
                    tapVisibility);
                historyColorSum += tapHistory.rgb * tapWeight;
                historyCoverageSum += tapCoverage * tapWeight;
                historyVisSum += tapVisibility * tapWeight;
                validWeight += tapWeight;
            }
            if (validWeight > 1.0e-4f)
            {
                const float invValid = 1.0f / validWeight;
                history = float4(historyColorSum * invValid,
                    historyCoverageSum * invValid);
                historyDirectVisibility = historyVisSum * invValid;
                historyConfidence = saturate(validWeight);
                historyValid = true;
                rejectionReason = validWeight < 0.999f ? tapReject : 0u;
            }
            else
            {
                rejectionReason = tapReject;
            }
        }
    }

    if (gGiTraceConstants.debugView == 14u)
    {
        gGiOutput[pixel] = pixel.x < width / 2u ? current : history;
        return;
    }
    if (gGiTraceConstants.debugView == 15u)
    {
        const float3 reasons[7] = {
            float3(0.0f, 0.8f, 0.1f), float3(0.8f, 0.0f, 0.8f),
            float3(0.0f, 0.2f, 1.0f), float3(1.0f, 0.1f, 0.0f),
            float3(1.0f, 0.65f, 0.0f), float3(0.0f, 0.8f, 0.8f),
            float3(1.0f, 1.0f, 1.0f) };
        gGiOutput[pixel] = float4(reasons[min(rejectionReason, 6u)], 1.0f);
        return;
    }
    if ((gGiTraceConstants.debugView >= 10u &&
         gGiTraceConstants.debugView <= 13u) ||
        gGiTraceConstants.debugView == 16u ||
        gGiTraceConstants.debugView == 17u ||
        gGiTraceConstants.debugView == 19u ||
        gGiTraceConstants.debugView == 20u)
    {
        gGiOutput[pixel] = gGiPrimaryInput.Load(int3(rawCenter, 0));
        return;
    }

    if ((gGiTraceConstants.giFlags & 1u) != 0u &&
        !historyValid && accepted > 0u)
        gGiDiagnostics.InterlockedAdd(108u, 1u);

    float appliedHistoryWeight = 0.0f;
    if (historyValid && current.a > 0.0f)
    {
        const float currentLuma = dot(max(current.rgb, 0.0f),
            float3(0.2126f, 0.7152f, 0.0722f));
        const float historyLuma = dot(max(history.rgb, 0.0f),
            float3(0.2126f, 0.7152f, 0.0722f));
        const float darkFactor = 1.0f - saturate(
            max(currentLuma, historyLuma) / 0.08f);
        const float sigmaScale = lerp(1.5f, 0.85f, darkFactor);
        const float3 clampRadius = max(sigma * sigmaScale,
            0.002f.xxx + abs(mean) * 0.05f);
        const float3 lo = max(mean - clampRadius, 0.0f);
        const float3 hi = mean + clampRadius;
        history.rgb = clamp(history.rgb, lo, hi);
        const float varianceLuma = dot(variance,
            float3(0.2126f, 0.7152f, 0.0722f));
        const float responsiveWeight = lerp(0.94f, 0.72f,
            saturate(varianceLuma * 8.0f));
        const float historyWeight = lerp(
            responsiveWeight, 0.97f, darkFactor) * historyConfidence;
        appliedHistoryWeight = historyWeight;
        current.rgb = lerp(current.rgb, history.rgb, historyWeight);
        currentDirectVisibility = lerp(currentDirectVisibility,
            historyDirectVisibility, historyWeight);
    }
    else if (historyValid && history.a > 0.0f)
    {
        // Preserve a validated history sample across a transient support hole,
        // but decay confidence so genuinely uncovered regions clear naturally.
        appliedHistoryWeight = 0.96f * historyConfidence;
        current.rgb = lerp(current.rgb, history.rgb, historyConfidence);
        current.a = history.a * 0.96f * historyConfidence;
        currentDirectVisibility = historyDirectVisibility;
    }
    if (gGiTraceConstants.debugView == 24u)
    {
        gGiOutput[pixel] = float4(appliedHistoryWeight.xxx, 1.0f);
        return;
    }
    if ((gGiTraceConstants.giFlags & 1u) != 0u)
    {
        const int traceSlot = giTracePixelSlot(pixel, uint2(width, height));
        if (traceSlot >= 0)
        {
            const uint traceAddress = giTracePixelAddress(traceSlot);
            uint4 traceState = gGiDiagnostics.Load4(traceAddress + 0u);
            traceState.w = rejectionReason;
            gGiDiagnostics.Store4(traceAddress + 0u, traceState);
            gGiDiagnostics.Store4(traceAddress + 96u,
                asuint(temporalCurrentInput));
            gGiDiagnostics.Store4(traceAddress + 112u, asuint(history));
            gGiDiagnostics.Store4(traceAddress + 128u, asuint(current));
            const float rawDirectVisibility = asfloat(
                gGiDiagnostics.Load(traceAddress + 144u));
            gGiDiagnostics.Store4(traceAddress + 144u, asuint(float4(
                rawDirectVisibility,
                appliedHistoryWeight, historyConfidence,
                currentDirectVisibility)));
        }
    }
    gGiOutput[pixel] = float4(current.rgb,
        giPackCoverageVisibility(current.a, currentDirectVisibility));
}

#endif
