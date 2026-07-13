// ic/core/coroutine_frame_pool.h
#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

namespace ic
{
    /*

        CoroutineFramePool — a tiny per-thread segregated free-list for coroutine
        frames.

        Why: every co_await that suspends and resumes on the job system would,
        by default, hit the global allocator for its coroutine frame. On the hot
        path (thousands of tasks per frame) that is both a throughput and a
        false-sharing hazard. Coroutine frames are small and extremely churny —
        allocated and freed constantly with a handful of recurring sizes — which
        is exactly the workload a segregated free-list eats for breakfast.

        Design:
        - Thread-local: each worker gets its own pool, so allocate/deallocate are
          lock-free and cache-friendly. A frame is nearly always freed on the
          same thread it was born on for lazily-started tasks; when it isn't, we
          simply fall back to freeing through the owning thread's global operator
          (see below), so cross-thread frees are correct, just not pooled.
        - Size-classed: sizes are rounded up to kGranularity and bucketed. Blocks
          carry a small header recording their rounded size class so deallocate
          can return them to the right bucket without the caller passing a size.
        - Bounded: each bucket caps its cached blocks; overflow is returned to the
          global allocator so idle pools don't pin unbounded memory.

        This is deliberately modest — correctness first, with the fast path being
        a single vector pop/push. promise_type opts in via operator new/delete
        (see task.h).

    */
    class CoroutineFramePool
    {
    public:
        static void* allocate(std::size_t size)
        {
            // Reserve room for the header and round up to a size class.
            const std::size_t total     = size + kHeaderSize;
            const std::size_t rounded   = roundUp(total);
            const std::size_t bucketIdx = bucketIndex(rounded);

            Pool& pool = tls();

            if (bucketIdx < kBucketCount)
            {
                std::vector<void*>& freeList = pool.buckets[bucketIdx];
                if (!freeList.empty())
                {
                    void* block = freeList.back();
                    freeList.pop_back();
                    return userPtr(block, rounded);
                }
            }

            // Miss (or oversized): fall back to the global allocator.
            void* block = ::operator new(rounded);
            return userPtr(block, rounded);
        }

        static void deallocate(void* userPointer) noexcept
        {
            if (!userPointer) return;

            Header*           header  = headerFor(userPointer);
            const std::size_t rounded = header->roundedSize;
            void*             block   = header;

            const std::size_t bucketIdx = bucketIndex(rounded);

            Pool& pool = tls();
            if (bucketIdx < kBucketCount &&
                pool.buckets[bucketIdx].size() < kMaxCachedPerBucket)
            {
                pool.buckets[bucketIdx].push_back(block);
                return;
            }

            ::operator delete(block);
        }

    private:
        // 16-byte granularity keeps user allocations naturally aligned for the
        // coroutine frame (which never needs more than max_align_t here).
        static constexpr std::size_t kGranularity      = 16;
        static constexpr std::size_t kHeaderSize       = 16;   // >= sizeof(Header), keeps alignment
        static constexpr std::size_t kBucketCount      = 32;   // covers rounded sizes up to 512 bytes
        static constexpr std::size_t kMaxCachedPerBucket = 256;

        struct Header
        {
            std::size_t roundedSize;
        };

        struct Pool
        {
            std::vector<void*> buckets[kBucketCount];

            ~Pool()
            {
                // Return every cached block to the global allocator on thread
                // exit so the pool never leaks.
                for (auto& bucket : buckets)
                {
                    for (void* block : bucket)
                    {
                        ::operator delete(block);
                    }
                }
            }
        };

        static Pool& tls()
        {
            thread_local Pool pool;
            return pool;
        }

        static std::size_t roundUp(std::size_t n) noexcept
        {
            return (n + (kGranularity - 1)) & ~(kGranularity - 1);
        }

        static std::size_t bucketIndex(std::size_t rounded) noexcept
        {
            return rounded / kGranularity;
        }

        // Lay out the block as [Header | user bytes]. The header records the
        // rounded size so deallocate can recover the size class the standard
        // sized-delete would otherwise hand us (coroutine frame delete is
        // unsized).
        static void* userPtr(void* block, std::size_t rounded) noexcept
        {
            auto* header        = static_cast<Header*>(block);
            header->roundedSize = rounded;
            return static_cast<void*>(static_cast<std::uint8_t*>(block) + kHeaderSize);
        }

        static Header* headerFor(void* userPointer) noexcept
        {
            return reinterpret_cast<Header*>(
                static_cast<std::uint8_t*>(userPointer) - kHeaderSize);
        }
    };
}
