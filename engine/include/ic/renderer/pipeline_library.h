#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <toml++/toml.hpp>

#include "ic/renderer/render_pipeline.h"
#include "ic/renderer/renderer_specification.h"

namespace ic
{
    class PipelineLibrary final
    {
    public:
        void load(const std::filesystem::path& path);
        void clear();

        bool contains(PipelineId id) const;

        GraphicsPipelineDesc resolveGraphics(
            PipelineId id,
            RendererBackendType backend,
            TextureFormat swapchainFormat) const;

        ComputePipelineDesc resolveCompute(
            PipelineId id,
            RendererBackendType backend) const;

    private:
        struct ShaderArtifacts
        {
            std::filesystem::path vertex;
            std::filesystem::path pixel;
        };

        struct ComputeShaderArtifacts
        {
            std::filesystem::path compute;
        };

        struct GraphicsPipelineTemplate
        {
            PipelineId id = {};
            GraphicsPipelineDesc desc = {};
            std::array<bool, 8> colorUsesSwapchain = {};
            std::unordered_map<RendererBackendType, ShaderArtifacts> shaders;
        };

        struct ComputePipelineTemplate
        {
            PipelineId id = {};
            ComputePipelineDesc desc = {};
            std::unordered_map<RendererBackendType, ComputeShaderArtifacts> shaders;
        };

        GraphicsPipelineTemplate parseGraphicsPipeline(
            std::string_view name,
            const toml::table& table) const;

        ComputePipelineTemplate parseComputePipeline(
            std::string_view name,
            const toml::table& table) const;

        static VertexLayoutKind parseVertexLayout(std::string_view value);
        static PipelineBindingLayoutKind parseBindingLayout(std::string_view value);
        static PrimitiveTopologyKind parseTopology(std::string_view value);
        static CullMode parseCullMode(std::string_view value);
        static CompareOp parseCompareOp(std::string_view value);
        static TextureFormat parseTextureFormat(std::string_view value);

        static std::string stringOr(
            const toml::table& table,
            std::string_view key,
            std::string_view fallback);

        static bool boolOr(
            const toml::table& table,
            std::string_view key,
            bool fallback);

        std::unordered_map<PipelineId, GraphicsPipelineTemplate, PipelineIdHash> m_graphics;
        std::unordered_map<PipelineId, ComputePipelineTemplate, PipelineIdHash> m_compute;
    };
}
