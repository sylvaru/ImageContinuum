#pragma once


namespace ic
{
    class AppBase;
    struct FrameContext;

    enum class FramePhase
    {
        Input,
        Simulation,
        RenderPrep,
        RenderSubmit
    };

    class FrameExecutor
    {
    public:
        explicit FrameExecutor(AppBase& app);

        void execute(FrameContext& frame);

    private:
        void runInput(FrameContext& frame);
        void runSimulation(FrameContext& frame);
        void runRenderPrep(FrameContext& frame);
        void runRenderSubmit(FrameContext& frame);

    private:
        AppBase& m_app;
    };
}