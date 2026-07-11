#include "ic/renderer/pipeline_library.h"

#include <stdexcept>

namespace
{
    std::filesystem::path resolveProjectPath(const std::filesystem::path& path)
    {
        if (path.is_absolute())
        {
            return path;
        }

        std::filesystem::path candidate = path;
        if (std::filesystem::exists(candidate))
        {
            return candidate;
        }

        std::filesystem::path cursor = std::filesystem::current_path();
        while (!cursor.empty())
        {
            candidate = (cursor / path).lexically_normal();
            if (std::filesystem::exists(candidate))
            {
                return candidate;
            }

            const std::filesystem::path parent = cursor.parent_path();
            if (parent == cursor)
            {
                break;
            }
            cursor = parent;
        }

        return path;
    }

    template<typename T>
    T requireEnum(
        std::string_view value,
        std::string_view field,
        T (*parser)(std::string_view))
    {
        try
        {
            return parser(value);
        }
        catch (const std::exception&)
        {
            throw std::runtime_error(
                "Invalid pipeline config value for " + std::string(field) +
                ": " + std::string(value));
        }
    }
}

namespace ic
{
    void PipelineLibrary::load(const std::filesystem::path& path)
    {
        clear();

        const std::filesystem::path resolvedPath =
            resolveProjectPath(path);

        if (!std::filesystem::exists(resolvedPath))
        {
            throw std::runtime_error(
                "Pipeline library file not found: " + path.string() +
                " (resolved: " + resolvedPath.string() + ")");
        }

        toml::table root{};
        try
        {
            root = toml::parse_file(resolvedPath.string());
        }
        catch (const toml::parse_error& e)
        {
            throw std::runtime_error(
                "Failed to parse pipeline library '" + path.string() +
                "' (resolved: " + resolvedPath.string() +
                "): " + std::string(e.description()));
        }

        const toml::table* pipelines =
            root["pipelines"].as_table();
        if (!pipelines)
        {
            throw std::runtime_error(
                "Pipeline library has no [pipelines] table: " +
                resolvedPath.string());
        }

        for (const auto& [key, node] : *pipelines)
        {
            const toml::table* pipelineTable = node.as_table();
            if (!pipelineTable)
            {
                continue;
            }

            const std::string name{ key.str() };
            const std::string type =
                stringOr(*pipelineTable, "type", "Graphics");

            if (type == "Graphics")
            {
                GraphicsPipelineTemplate pipeline =
                    parseGraphicsPipeline(name, *pipelineTable);
                m_graphics.emplace(pipeline.id, std::move(pipeline));
            }
            else if (type == "Compute")
            {
                ComputePipelineTemplate pipeline =
                    parseComputePipeline(name, *pipelineTable);
                m_compute.emplace(pipeline.id, std::move(pipeline));
            }
        }
    }

    void PipelineLibrary::clear()
    {
        m_graphics.clear();
        m_compute.clear();
    }

    bool PipelineLibrary::contains(PipelineId id) const
    {
        return m_graphics.find(id) != m_graphics.end();
    }

    GraphicsPipelineDesc PipelineLibrary::resolveGraphics(
        PipelineId id,
        RendererBackendType backend,
        TextureFormat swapchainFormat) const
    {
        auto it = m_graphics.find(id);
        if (it == m_graphics.end())
        {
            throw std::runtime_error("Unknown graphics pipeline id.");
        }

        const GraphicsPipelineTemplate& pipeline = it->second;
        auto shaderIt = pipeline.shaders.find(backend);
        if (shaderIt == pipeline.shaders.end())
        {
            throw std::runtime_error(
                "Graphics pipeline has no shader artifacts for backend.");
        }

        GraphicsPipelineDesc desc = pipeline.desc;
        desc.shaders.vertexShader = shaderIt->second.vertex;
        desc.shaders.pixelShader = shaderIt->second.pixel;

        for (uint32_t i = 0; i < desc.colorAttachmentCount; ++i)
        {
            if (pipeline.colorUsesSwapchain[i])
            {
                desc.colorFormats[i] = swapchainFormat;
            }
        }

        return desc;
    }

    ComputePipelineDesc PipelineLibrary::resolveCompute(
        PipelineId id,
        RendererBackendType backend) const
    {
        auto it = m_compute.find(id);
        if (it == m_compute.end())
        {
            throw std::runtime_error("Unknown compute pipeline id.");
        }

        const ComputePipelineTemplate& pipeline = it->second;
        auto shaderIt = pipeline.shaders.find(backend);
        if (shaderIt == pipeline.shaders.end())
        {
            throw std::runtime_error(
                "Compute pipeline has no shader artifacts for backend.");
        }

        ComputePipelineDesc desc = pipeline.desc;
        desc.shaders.computeShader = shaderIt->second.compute;
        return desc;
    }

    PipelineLibrary::GraphicsPipelineTemplate
        PipelineLibrary::parseGraphicsPipeline(
            std::string_view name,
            const toml::table& table) const
    {
        GraphicsPipelineTemplate pipeline{};
        pipeline.id = makePipelineId(name);
        pipeline.desc.debugName = std::string(name);
        pipeline.desc.vertexLayout =
            requireEnum(
                stringOr(table, "vertexLayout", "Unknown"),
                "vertexLayout",
                parseVertexLayout);
        pipeline.desc.bindingLayout =
            requireEnum(
                stringOr(table, "bindingLayout", "Unknown"),
                "bindingLayout",
                parseBindingLayout);
        pipeline.desc.topology =
            requireEnum(
                stringOr(table, "topology", "TriangleList"),
                "topology",
                parseTopology);

        if (const toml::table* raster = table["raster"].as_table())
        {
            pipeline.desc.raster.cullMode =
                requireEnum(
                    stringOr(*raster, "cullMode", "Back"),
                    "raster.cullMode",
                    parseCullMode);
            pipeline.desc.raster.depthClip =
                boolOr(*raster, "depthClip", true);
        }

        if (const toml::table* depth = table["depth"].as_table())
        {
            pipeline.desc.depth.enabled =
                boolOr(*depth, "enabled", false);
            pipeline.desc.depth.write =
                boolOr(*depth, "write", false);
            pipeline.desc.depth.compare =
                requireEnum(
                    stringOr(*depth, "compare", "LessEqual"),
                    "depth.compare",
                    parseCompareOp);
            pipeline.desc.depth.format =
                requireEnum(
                    stringOr(*depth, "format", "Unknown"),
                    "depth.format",
                    parseTextureFormat);
        }

        if (const toml::table* blend = table["blend"].as_table())
        {
            pipeline.desc.blend.enabled =
                boolOr(*blend, "enabled", false);
        }

        if (const toml::table* color = table["color"].as_table())
        {
            if (const toml::array* formats = color->get_as<toml::array>("formats"))
            {
                uint32_t index = 0;
                for (const toml::node& node : *formats)
                {
                    if (index >= pipeline.desc.colorFormats.size())
                    {
                        break;
                    }

                    const std::string value =
                        node.value_or<std::string>("");
                    if (value == "Swapchain")
                    {
                        pipeline.colorUsesSwapchain[index] = true;
                        pipeline.desc.colorFormats[index] =
                            TextureFormat::Unknown;
                    }
                    else
                    {
                        pipeline.desc.colorFormats[index] =
                            requireEnum(
                                value,
                                "color.formats",
                                parseTextureFormat);
                    }
                    ++index;
                }
                pipeline.desc.colorAttachmentCount = index;
            }
        }

        if (const toml::table* shaders = table["shaders"].as_table())
        {
            if (const toml::table* dx12 = shaders->get_as<toml::table>("dx12"))
            {
#ifdef _WIN32
                pipeline.shaders.emplace(
                    RendererBackendType::DX12,
                    ShaderArtifacts{
                        .vertex = stringOr(*dx12, "vertex", ""),
                        .pixel = stringOr(*dx12, "pixel", "")
                    });
#endif
            }

            if (const toml::table* vulkan = shaders->get_as<toml::table>("vulkan"))
            {
                pipeline.shaders.emplace(
                    RendererBackendType::Vulkan,
                    ShaderArtifacts{
                        .vertex = stringOr(*vulkan, "vertex", ""),
                        .pixel = stringOr(*vulkan, "pixel", "")
                    });
            }
        }

        if (pipeline.desc.colorAttachmentCount == 0)
        {
            const bool depthOnly =
                pipeline.desc.depth.enabled &&
                pipeline.desc.depth.format != TextureFormat::Unknown;
            if (!depthOnly)
            {
                throw std::runtime_error(
                    "Graphics pipeline requires color or depth output.");
            }
        }

        return pipeline;
    }

    PipelineLibrary::ComputePipelineTemplate
        PipelineLibrary::parseComputePipeline(
            std::string_view name,
            const toml::table& table) const
    {
        ComputePipelineTemplate pipeline{};
        pipeline.id = makePipelineId(name);
        pipeline.desc.debugName = std::string(name);
        pipeline.desc.bindingLayout =
            requireEnum(
                stringOr(table, "bindingLayout", "Empty"),
                "bindingLayout",
                parseBindingLayout);

        if (const toml::table* shaders = table["shaders"].as_table())
        {
            if (const toml::table* dx12 = shaders->get_as<toml::table>("dx12"))
            {
#ifdef _WIN32
                pipeline.shaders.emplace(
                    RendererBackendType::DX12,
                    ComputeShaderArtifacts{
                        .compute = stringOr(*dx12, "compute", "")
                    });
#endif
            }

            if (const toml::table* vulkan = shaders->get_as<toml::table>("vulkan"))
            {
                pipeline.shaders.emplace(
                    RendererBackendType::Vulkan,
                    ComputeShaderArtifacts{
                        .compute = stringOr(*vulkan, "compute", "")
                    });
            }
        }

        return pipeline;
    }

    VertexLayoutKind PipelineLibrary::parseVertexLayout(std::string_view value)
    {
        if (value == "AssetVertex") return VertexLayoutKind::AssetVertex;
        if (value == "Unknown") return VertexLayoutKind::Unknown;
        throw std::runtime_error("Invalid vertex layout.");
    }

    PipelineBindingLayoutKind PipelineLibrary::parseBindingLayout(
        std::string_view value)
    {
        if (value == "ForwardBindless")
            return PipelineBindingLayoutKind::ForwardBindless;
        if (value == "ComputeStorageBuffer")
            return PipelineBindingLayoutKind::ComputeStorageBuffer;
        if (value == "EnvironmentConvert")
            return PipelineBindingLayoutKind::EnvironmentConvert;
        if (value == "Skybox")
            return PipelineBindingLayoutKind::Skybox;
        if (value == "ClusteredForward")
            return PipelineBindingLayoutKind::ClusteredForward;
        if (value == "HiZDepthPyramid")
            return PipelineBindingLayoutKind::HiZDepthPyramid;
        if (value == "GpuFrustumCull")
            return PipelineBindingLayoutKind::GpuFrustumCull;
        if (value == "PathTrace")
            return PipelineBindingLayoutKind::PathTrace;
        if (value == "PathTraceTonemap")
            return PipelineBindingLayoutKind::PathTraceTonemap;
        if (value == "Empty")
            return PipelineBindingLayoutKind::Empty;
        if (value == "Unknown")
            return PipelineBindingLayoutKind::Unknown;
        throw std::runtime_error("Invalid binding layout.");
    }

    PrimitiveTopologyKind PipelineLibrary::parseTopology(std::string_view value)
    {
        if (value == "TriangleList")
            return PrimitiveTopologyKind::TriangleList;
        throw std::runtime_error("Invalid topology.");
    }

    CullMode PipelineLibrary::parseCullMode(std::string_view value)
    {
        if (value == "None") return CullMode::None;
        if (value == "Front") return CullMode::Front;
        if (value == "Back") return CullMode::Back;
        throw std::runtime_error("Invalid cull mode.");
    }

    CompareOp PipelineLibrary::parseCompareOp(std::string_view value)
    {
        if (value == "Never") return CompareOp::Never;
        if (value == "Less") return CompareOp::Less;
        if (value == "Equal") return CompareOp::Equal;
        if (value == "LessEqual") return CompareOp::LessEqual;
        if (value == "Greater") return CompareOp::Greater;
        if (value == "Always") return CompareOp::Always;
        throw std::runtime_error("Invalid compare op.");
    }

    TextureFormat PipelineLibrary::parseTextureFormat(std::string_view value)
    {
        if (value == "RGBA8_UNorm") return TextureFormat::RGBA8_UNorm;
        if (value == "RGBA8_SRGB") return TextureFormat::RGBA8_SRGB;
        if (value == "RGBA32_Float") return TextureFormat::RGBA32_Float;
        if (value == "BGRA8_UNorm") return TextureFormat::BGRA8_UNorm;
        if (value == "BGRA8_SRGB") return TextureFormat::BGRA8_SRGB;
        if (value == "D32_Float") return TextureFormat::D32_Float;
        if (value == "R32_Float") return TextureFormat::R32_Float;
        if (value == "Unknown") return TextureFormat::Unknown;
        throw std::runtime_error("Invalid texture format.");
    }

    std::string PipelineLibrary::stringOr(
        const toml::table& table,
        std::string_view key,
        std::string_view fallback)
    {
        return table[key].value_or<std::string>(std::string(fallback));
    }

    bool PipelineLibrary::boolOr(
        const toml::table& table,
        std::string_view key,
        bool fallback)
    {
        if (const std::optional<bool> value = table[key].value<bool>())
        {
            return *value;
        }
        return fallback;
    }
}
