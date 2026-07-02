#include "ic/renderer/path_tracing/path_trace_scene_builder.h"

#include "ic/core/asset_manager.h"
#include "ic/scene/scene_render_view.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <glm/gtc/matrix_inverse.hpp>

namespace ic
{
    namespace
    {
        struct Bounds
        {
            glm::vec3 min =
                glm::vec3(std::numeric_limits<float>::max());
            glm::vec3 max =
                glm::vec3(std::numeric_limits<float>::lowest());
        };

        struct BuildTriangle
        {
            PathTraceTriangle triangle;
            Bounds bounds;
            glm::vec3 centroid = glm::vec3(0.0f);
        };

        void expand(Bounds& bounds, const glm::vec3& p)
        {
            bounds.min = glm::min(bounds.min, p);
            bounds.max = glm::max(bounds.max, p);
        }

        void expand(Bounds& bounds, const Bounds& other)
        {
            bounds.min = glm::min(bounds.min, other.min);
            bounds.max = glm::max(bounds.max, other.max);
        }

        uint32_t largestAxis(const glm::vec3& extent)
        {
            if (extent.x >= extent.y && extent.x >= extent.z)
            {
                return 0;
            }

            return extent.y >= extent.z ? 1u : 2u;
        }

        float axisValue(const glm::vec3& v, uint32_t axis)
        {
            return axis == 0 ? v.x : axis == 1 ? v.y : v.z;
        }

        uint32_t buildBvhRecursive(
            std::vector<BuildTriangle>& triangles,
            uint32_t first,
            uint32_t count,
            std::vector<PathTraceBVHNode>& nodes,
            uint32_t nodeIndex)
        {
            Bounds bounds{};
            Bounds centroidBounds{};
            for (uint32_t i = 0; i < count; ++i)
            {
                const BuildTriangle& tri = triangles[first + i];
                expand(bounds, tri.bounds);
                expand(centroidBounds, tri.centroid);
            }

            nodes[nodeIndex].boundsMin = bounds.min;
            nodes[nodeIndex].boundsMax = bounds.max;

            constexpr uint32_t maxLeafTriangles = 4;
            if (count <= maxLeafTriangles)
            {
                nodes[nodeIndex].leftFirst = first;
                nodes[nodeIndex].count = count;
                return nodeIndex;
            }

            const glm::vec3 centroidExtent =
                centroidBounds.max - centroidBounds.min;
            const uint32_t axis = largestAxis(centroidExtent);
            const uint32_t mid = first + count / 2;

            std::nth_element(
                triangles.begin() + first,
                triangles.begin() + mid,
                triangles.begin() + first + count,
                [axis](const BuildTriangle& a, const BuildTriangle& b)
                {
                    return axisValue(a.centroid, axis) <
                        axisValue(b.centroid, axis);
                });

            const uint32_t leftIndex =
                static_cast<uint32_t>(nodes.size());
            nodes.push_back({});
            const uint32_t rightIndex =
                static_cast<uint32_t>(nodes.size());
            nodes.push_back({});

            nodes[nodeIndex].leftFirst = leftIndex;
            nodes[nodeIndex].count = 0;
            buildBvhRecursive(
                triangles,
                first,
                mid - first,
                nodes,
                leftIndex);
            buildBvhRecursive(
                triangles,
                mid,
                first + count - mid,
                nodes,
                rightIndex);

            return nodeIndex;
        }
    }

    PathTraceSceneData buildPathTraceSceneData(
        const SceneRenderView& scene,
        const AssetManager& assets)
    {
        PathTraceSceneData data{};
        std::vector<BuildTriangle> buildTriangles;

        for (const SceneModelRenderItem& item : scene.models)
        {
            const ModelAsset* model = assets.model(item.model);
            if (!model || model->vertices.empty() || model->indices.empty())
            {
                continue;
            }

            const uint32_t materialOffset =
                static_cast<uint32_t>(data.materials.size());
            data.materials.reserve(
                data.materials.size() +
                std::max<size_t>(1, model->materials.size()));

            for (const MaterialAsset& src : model->materials)
            {
                PathTraceMaterial dst{};
                dst.baseColor = src.baseColorFactor;
                dst.emissive =
                    glm::vec4(src.emissiveFactor, 0.0f);
                if (src.name.find("Light") != std::string::npos ||
                    src.name.find("light") != std::string::npos)
                {
                    const glm::vec3 color =
                        glm::vec3(src.baseColorFactor);
                    dst.emissive =
                        glm::vec4(glm::max(color, glm::vec3(1.0f)) * 18.0f, 0.0f);
                }
                dst.roughness = src.roughnessFactor;
                dst.metallic = src.metallicFactor;
                data.materials.push_back(dst);
            }

            if (model->materials.empty())
            {
                PathTraceMaterial material{};
                material.baseColor = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
                material.roughness = 1.0f;
                material.metallic = 0.0f;
                data.materials.push_back(material);
            }

            std::vector<glm::mat4> meshNodeTransforms(
                model->meshes.size(),
                glm::mat4(1.0f));

            for (const NodeAsset& node : model->nodes)
            {
                if (node.meshIndex >= 0 &&
                    static_cast<size_t>(node.meshIndex) <
                        meshNodeTransforms.size())
                {
                    meshNodeTransforms[
                        static_cast<size_t>(node.meshIndex)] =
                        node.worldMatrix;
                }
            }

            for (uint32_t meshIndex = 0;
                 meshIndex < model->meshes.size();
                 ++meshIndex)
            {
                const MeshAsset& mesh = model->meshes[meshIndex];
                const glm::mat4 world =
                    item.world * meshNodeTransforms[meshIndex];
                const glm::mat3 normalWorld =
                    glm::inverseTranspose(glm::mat3(world));

                for (const MeshPrimitiveAsset& primitive : mesh.primitives)
                {
                    const uint32_t materialIndex =
                        materialOffset +
                        (primitive.materialIndex != UINT32_MAX &&
                            primitive.materialIndex < model->materials.size()
                            ? primitive.materialIndex
                            : 0u);

                    for (uint32_t i = 0;
                         i + 2 < primitive.indexCount;
                         i += 3)
                    {
                        const uint32_t srcIndices[3] =
                        {
                            model->indices[primitive.firstIndex + i + 0],
                            model->indices[primitive.firstIndex + i + 1],
                            model->indices[primitive.firstIndex + i + 2]
                        };

                        if (srcIndices[0] >= model->vertices.size() ||
                            srcIndices[1] >= model->vertices.size() ||
                            srcIndices[2] >= model->vertices.size())
                        {
                            continue;
                        }

                        const uint32_t baseVertex =
                            static_cast<uint32_t>(data.vertices.size());

                        Bounds triBounds{};
                        glm::vec3 centroid(0.0f);

                        for (uint32_t v = 0; v < 3; ++v)
                        {
                            const AssetVertex& src =
                                model->vertices[srcIndices[v]];
                            PathTraceVertex dst{};
                            const glm::vec3 position =
                                glm::vec3(world * glm::vec4(src.position, 1.0f));
                            glm::vec3 normal = normalWorld * src.normal;
                            if (glm::dot(normal, normal) >
                                1.0e-8f)
                            {
                                normal = glm::normalize(normal);
                            }

                            if (!std::isfinite(normal.x) ||
                                !std::isfinite(normal.y) ||
                                !std::isfinite(normal.z))
                            {
                                normal = glm::vec3(0.0f, 1.0f, 0.0f);
                            }

                            dst.position = glm::vec4(position, 1.0f);
                            dst.normal = glm::vec4(normal, 0.0f);

                            data.vertices.push_back(dst);
                            expand(triBounds, position);
                            centroid += position;
                        }

                        BuildTriangle tri{};
                        tri.triangle.i0 = baseVertex + 0;
                        tri.triangle.i1 = baseVertex + 1;
                        tri.triangle.i2 = baseVertex + 2;
                        tri.triangle.materialIndex = materialIndex;
                        tri.bounds = triBounds;
                        tri.centroid = centroid / 3.0f;
                        buildTriangles.push_back(tri);
                    }
                }
            }
        }

        if (buildTriangles.empty())
        {
            data.vertices.clear();
            data.materials.clear();
            return data;
        }

        data.bvhNodes.reserve(buildTriangles.size() * 2);
        data.bvhNodes.push_back({});
        buildBvhRecursive(
            buildTriangles,
            0,
            static_cast<uint32_t>(buildTriangles.size()),
            data.bvhNodes,
            0);

        data.triangles.reserve(buildTriangles.size());
        for (const BuildTriangle& tri : buildTriangles)
        {
            data.triangles.push_back(tri.triangle);
        }

        return data;
    }
}
