#include "ic/common/ic_pch.h"
#include "ic/renderer/renderer_gpu_assets.h"

#include "ic/core/frame_context.h"

namespace ic
{
    namespace
    {
        constexpr uint32_t MaterialFlagDoubleSided = 1u << 0u;
        constexpr uint32_t MaterialFlagAlphaBlend = 1u << 1u;
        constexpr uint32_t MaterialFlagAlphaMask = 1u << 2u;
        constexpr uint32_t MaterialFlagUnlit = 1u << 3u;
    }

    GpuMaterialData makeGpuMaterialData(const MaterialAsset& src)
    {
        GpuMaterialData dst{};
        dst.baseColorFactor = src.baseColorFactor;
        dst.metallicFactor = src.metallicFactor;
        dst.roughnessFactor = src.roughnessFactor;
        dst.alphaCutoff = src.alphaCutoff;
        dst.flags =
            (src.doubleSided ? MaterialFlagDoubleSided : 0u) |
            (src.alphaBlend ? MaterialFlagAlphaBlend : 0u) |
            (src.alphaMask ? MaterialFlagAlphaMask : 0u) |
            (src.unlit ? MaterialFlagUnlit : 0u);

        return dst;
    }

    GpuModelHandle RendererGpuAssetCache::requestModel(AssetHandle model)
    {
        if (!model)
        {
            return {};
        }

        if (auto it = m_modelLookup.find(model); it != m_modelLookup.end())
        {
            return it->second;
        }

        GpuModel gpuModel{};
        gpuModel.handle.index = static_cast<uint32_t>(m_models.size());
        gpuModel.handle.generation = 1;
        gpuModel.source = model;

        const GpuModelHandle handle = gpuModel.handle;
        m_modelLookup.emplace(model, handle);
        m_models.push_back(std::move(gpuModel));

        return handle;
    }

    void RendererGpuAssetCache::updateUploads(
        [[maybe_unused]] FrameContext& frame,
        AssetManager& assets)
    {
        for (GpuModel& gpuModel : m_models)
        {
            if (gpuModel.uploaded || gpuModel.uploadQueued)
            {
                continue;
            }

            const ModelAsset* modelAsset = assets.model(gpuModel.source);
            if (!modelAsset)
            {
                continue;
            }

            gpuModel.materials.clear();
            gpuModel.materials.reserve(modelAsset->materials.size());

            for (const MaterialAsset& src : modelAsset->materials)
            {
                gpuModel.materials.push_back(makeGpuMaterialData(src));
            }

            if (gpuModel.materials.empty())
            {
                gpuModel.materials.push_back(GpuMaterialData{});
            }

            gpuModel.meshes.clear();
            for (const MeshAsset& mesh : modelAsset->meshes)
            {
                for (const MeshPrimitiveAsset& primitive : mesh.primitives)
                {
                    GpuMesh gpuMesh{};
                    gpuMesh.materialIndex =
                        primitive.materialIndex != UINT32_MAX
                            ? primitive.materialIndex
                            : 0u;
                    gpuMesh.indexCount = primitive.indexCount;
                    gpuMesh.firstIndex = primitive.firstIndex;
                    gpuMesh.vertexOffset = primitive.firstVertex;
                    gpuModel.meshes.push_back(gpuMesh);
                }
            }

            gpuModel.uploadQueued = true;
        }
    }

    const GpuModel* RendererGpuAssetCache::model(GpuModelHandle handle) const
    {
        if (!handle || handle.index >= m_models.size())
        {
            return nullptr;
        }

        const GpuModel& model = m_models[handle.index];
        return model.handle.generation == handle.generation ? &model : nullptr;
    }
}
