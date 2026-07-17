#include "ic/renderer/gpu_queue_profiler.h"

#include <algorithm>
#include <limits>

namespace ic
{
    namespace
    {
        // Sort + coalesce touching/overlapping intervals so a queue's busy time
        // is measured as set length, never as a sum of possibly-overlapping
        // pass durations.
        void mergeIntervals(std::vector<GpuPassSample>& intervals)
        {
            if (intervals.empty())
            {
                return;
            }

            std::sort(
                intervals.begin(),
                intervals.end(),
                [](const GpuPassSample& a, const GpuPassSample& b)
                {
                    return a.beginMs < b.beginMs;
                });

            size_t write = 0;
            for (size_t read = 1; read < intervals.size(); ++read)
            {
                if (intervals[read].beginMs <= intervals[write].endMs)
                {
                    intervals[write].endMs =
                        std::max(intervals[write].endMs, intervals[read].endMs);
                }
                else
                {
                    intervals[++write] = intervals[read];
                }
            }
            intervals.resize(write + 1);
        }

        double totalLength(const std::vector<GpuPassSample>& intervals)
        {
            double total = 0.0;
            for (const GpuPassSample& interval : intervals)
            {
                total += interval.endMs - interval.beginMs;
            }
            return total;
        }

        // Two-pointer sweep over two already-merged interval sets.
        double intersectionLength(
            const std::vector<GpuPassSample>& a,
            const std::vector<GpuPassSample>& b)
        {
            double total = 0.0;
            size_t i = 0;
            size_t j = 0;
            while (i < a.size() && j < b.size())
            {
                const double begin = std::max(a[i].beginMs, b[j].beginMs);
                const double end = std::min(a[i].endMs, b[j].endMs);
                if (end > begin)
                {
                    total += end - begin;
                }

                if (a[i].endMs < b[j].endMs)
                {
                    ++i;
                }
                else
                {
                    ++j;
                }
            }
            return total;
        }
    }

    GpuQueueTimelineStats analyzeGpuQueueTimeline(
        std::span<const GpuPassSample> samples)
    {
        std::vector<GpuPassSample> graphics;
        std::vector<GpuPassSample> compute;
        return analyzeGpuQueueTimeline(samples, graphics, compute);
    }

    GpuQueueTimelineStats analyzeGpuQueueTimeline(
        std::span<const GpuPassSample> samples,
        std::vector<GpuPassSample>& graphics,
        std::vector<GpuPassSample>& compute)
    {
        GpuQueueTimelineStats stats{};
        graphics.clear();
        compute.clear();
        double firstBegin = std::numeric_limits<double>::max();
        double lastEnd = std::numeric_limits<double>::lowest();

        for (const GpuPassSample& sample : samples)
        {
            // Drop degenerate/inverted samples rather than letting a bad
            // readback silently inflate busy time.
            if (!(sample.endMs > sample.beginMs))
            {
                continue;
            }

            firstBegin = std::min(firstBegin, sample.beginMs);
            lastEnd = std::max(lastEnd, sample.endMs);

            switch (sample.queue)
            {
            case QueueType::Graphics:
                graphics.push_back(sample);
                break;
            case QueueType::Compute:
                compute.push_back(sample);
                break;
            case QueueType::Transfer:
                break;
            }
        }

        if (graphics.empty() && compute.empty())
        {
            return stats;
        }

        mergeIntervals(graphics);
        mergeIntervals(compute);

        stats.graphicsBusyMs = totalLength(graphics);
        stats.computeBusyMs = totalLength(compute);
        stats.overlapMs = intersectionLength(graphics, compute);
        stats.spanMs = lastEnd - firstBegin;
        stats.valid = true;
        return stats;
    }
}
