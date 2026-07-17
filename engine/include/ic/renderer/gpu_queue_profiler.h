// ic/renderer/gpu_queue_profiler.h
#pragma once
#include <cstdint>
#include <span>
#include <vector>

#include "frame_graph/frame_graph_types.h"

namespace ic
{
    // One pass's GPU execution interval, resolved from timestamp queries and
    // converted into a single CPU-clock (millisecond) domain that is shared by
    // every queue. Converting into a common domain is what makes cross-queue
    // comparison meaningful: D3D12 reports a separate timestamp frequency per
    // queue, so raw ticks from the graphics and compute queues are NOT directly
    // comparable and must each be mapped through that queue's own calibration.
    struct GpuPassSample
    {
        GraphNodeId node{};
        QueueType queue = QueueType::Graphics;
        double beginMs = 0.0;
        double endMs = 0.0;
    };

    // Measured (not inferred) queue-occupancy summary for one frame.
    //
    // The distinction this exists to draw: a pass being *scheduled* on the
    // compute queue says nothing about whether it *ran concurrently* with
    // graphics work. Only timestamps can answer that, and overlapMs is the
    // answer. A frame where computeBusyMs > 0 but overlapMs ~= 0 means async
    // compute moved work to another queue and bought nothing -- it only added
    // submission and synchronization cost.
    struct GpuQueueTimelineStats
    {
        // Union (not sum) of each queue's busy intervals, so passes that run
        // concurrently on one queue are not double counted.
        double graphicsBusyMs = 0.0;
        double computeBusyMs = 0.0;

        // Wall-clock length of the intersection of the graphics-busy set and
        // the compute-busy set: the time both queues had work executing at
        // once. This is the only honest definition of "async compute worked".
        double overlapMs = 0.0;

        // First begin to last end across all queues.
        double spanMs = 0.0;

        bool valid = false;

        // Fraction of the compute queue's work that actually ran alongside
        // graphics. 1.0 = fully hidden behind graphics; 0.0 = fully exposed
        // (serialized), which is strictly worse than staying on graphics.
        [[nodiscard]] double overlapFraction() const
        {
            return computeBusyMs > 0.0 ? overlapMs / computeBusyMs : 0.0;
        }
    };

    // Merges each queue's intervals and intersects graphics against compute.
    // Pure function over samples: no GPU or backend state, so it is unit
    // testable and shared by both backends.
    [[nodiscard]] GpuQueueTimelineStats analyzeGpuQueueTimeline(
        std::span<const GpuPassSample> samples);

    // Allocation-free overload for frame loops. The caller owns reusable
    // scratch vectors; their contents are overwritten.
    [[nodiscard]] GpuQueueTimelineStats analyzeGpuQueueTimeline(
        std::span<const GpuPassSample> samples,
        std::vector<GpuPassSample>& graphicsScratch,
        std::vector<GpuPassSample>& computeScratch);

    // Shape of the compiled graph's submission schedule. Async compute is not
    // free on the CPU: the compiler emits one batch per (execution level,
    // queue), and every batch costs a submit plus a fence signal, with
    // cross-queue edges adding waits. When the frame is CPU-bound, that cost is
    // paid directly out of the frame time while the overlap it buys is spent on
    // a GPU that already had idle capacity.
    struct FrameGraphTopology
    {
        uint32_t passes = 0;
        uint32_t levels = 0;
        uint32_t batches = 0;
        uint32_t computeBatches = 0;
        uint32_t crossQueueWaits = 0;
    };
}
