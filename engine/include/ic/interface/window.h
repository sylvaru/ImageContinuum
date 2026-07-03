// ic/interface/window.h
#pragma once

namespace ic
{
    struct Event;

    enum class WindowMode
    {
        Windowed,
        Maximized,
        BorderlessFullscreen,
        Fullscreen
    };

    struct WindowSpecification
    {
        std::string title = "ImageContinuum";

        uint32_t width = 1920;
        uint32_t height = 1080;

        WindowMode mode = WindowMode::Windowed;
        bool resizable = true;
        bool maximized = false;
        bool fullscreen = false;
    };

    using EventCallbackFn = std::function<void(Event)>;

    class Window
    {
    public:
        virtual ~Window() = default;

        virtual void pollEvents() = 0;

        virtual bool shouldClose() const = 0;

        virtual uint32_t getWidth() const = 0;
        virtual uint32_t getHeight() const = 0;

        virtual void* getNativeHandle() const = 0;

        virtual void setTitle(std::string_view title) = 0;

        virtual void requestClose() = 0;

        virtual void bindEventSink(EventCallbackFn fn) = 0;
    };
}
