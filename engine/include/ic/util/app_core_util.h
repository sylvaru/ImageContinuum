#pragma once

#include <chrono>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace ic
{
    // Blocks the calling thread for `seconds` accurately and without
    // busy-waiting. On Windows this uses a high-resolution waitable timer
    // (Win10 1803+) so the wait is precise to well under a millisecond
    // WITHOUT globally raising the system timer resolution (which would hurt
    // power use); it falls back to std::this_thread::sleep_for otherwise.
    inline void preciseSleepSeconds(double seconds)
    {
        if (seconds <= 0.0)
        {
            return;
        }
#ifdef _WIN32
#ifdef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
        static HANDLE timer = CreateWaitableTimerExW(
            nullptr, nullptr,
            CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
            TIMER_ALL_ACCESS);
        if (timer)
        {
            LARGE_INTEGER due{};
            due.QuadPart = -static_cast<LONGLONG>(seconds * 1.0e7);
            if (SetWaitableTimerEx(
                timer, &due, 0, nullptr, nullptr, nullptr, 0))
            {
                WaitForSingleObject(timer, INFINITE);
                return;
            }
        }
#endif
#endif
        std::this_thread::sleep_for(
            std::chrono::duration<double>(seconds));
    }
}
