// engine/include/image_continuum/interface/window.h
#pragma once



namespace ic
{
    class Window
    {
    public:
        virtual ~Window() = default;

        virtual void pollEvents() = 0;

        virtual bool shouldClose() const = 0;

        virtual uint32_t getWidth() const = 0;
        virtual uint32_t getHeight() const = 0;

        virtual void* getNativeHandle() const = 0;
    };
}