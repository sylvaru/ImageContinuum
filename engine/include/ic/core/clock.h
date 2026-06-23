// ic/core/clock.h
#pragma once
#include <chrono>

namespace ic 
{

    class Clock
    {
    public:
        Clock()
        {
            reset();
        }

        void reset() 
        {
            m_startTime = std::chrono::steady_clock::now();
            m_lastTime = m_startTime;
        }

        // Ticks the clock, calculating delta time since the last frame
        float tick() 
        {
            auto currentTime = std::chrono::steady_clock::now();

            // Calculate delta time in seconds (as a float for transient step)
            std::chrono::duration<float> delta = currentTime - m_lastTime;
            m_lastTime = currentTime;

            return delta.count();
        }

        // Returns total uptime in seconds since reset() was called
        float getTimeSinceStart() const 
        {
            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<float> total = currentTime - m_startTime;
            return total.count();
        }

    private:
        std::chrono::steady_clock::time_point m_startTime;
        std::chrono::steady_clock::time_point m_lastTime;
    };

}
