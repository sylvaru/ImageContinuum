#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <concurrentqueue.h>
#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "ic/core/asset_manager.h"
#include "ic/core/frame_context.h"
#include "ic/core/job_system.h"
#include "ic/scene/scene.h"
#include "ic/scene/scene_entity.h"
#include "ic/scene/scene_render_view.h"
#include "ic/scene/scene_types.h"

namespace ic
{
    class SceneManager final
    {
    public:
        SceneManager() = default;
        ~SceneManager();

        SceneManager(const SceneManager&) = delete;
        SceneManager& operator=(const SceneManager&) = delete;

        void init(
            SceneManagerDesc desc,
            AssetManager& assets,
            JobSystem& jobs);

        void shutdown();

        void update(FrameContext& frame);

        SceneHandle createEmptyScene(std::string name);

        SceneHandle loadSceneAsync(
            std::filesystem::path path,
            SceneLoadOptions options = {});

        void unloadScene(SceneHandle handle);
        void setActiveScene(SceneHandle handle);

        Scene* activeScene();
        const Scene* activeScene() const;

        entt::registry* activeRegistry();
        const entt::registry* activeRegistry() const;

        const SceneRenderView& renderView() const
        {
            return m_renderView;
        }

        SceneState sceneState(SceneHandle handle) const;
        std::string sceneError(SceneHandle handle) const;

        SceneEntity createEntity(std::string_view name = {});
        void destroyEntity(SceneEntity entity);

        SceneEntity instantiateModel(
            AssetHandle model,
            std::string_view name = {});

    private:
        struct SceneRecord
        {
            SceneHandle handle = {};
            SceneState state = SceneState::Empty;

            std::unique_ptr<Scene> scene;

            std::filesystem::path sourcePath;
            std::string error;

            JobCounter counter;
        };

        struct LoadedSceneData
        {
            std::string name;
            std::filesystem::path sourcePath;
            AssetHandle environmentTexture = {};
            EnvironmentSettings environmentSettings = {};

            struct EntityDesc
            {
                uint64_t id = 0;
                std::string name;

                glm::vec3 translation = glm::vec3(0.0f);
                glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
                glm::vec3 scale = glm::vec3(1.0f);

                int64_t parentId = -1;

                std::filesystem::path modelPath;
                AssetHandle modelHandle = {};

                bool hasCamera = false;
                CameraComponent camera = {};

                bool hasFlyCamera = false;
                FlyCameraControllerComponent flyCamera = {};

                bool hasLight = false;
                LightComponent light = {};
            };

            std::vector<EntityDesc> entities;
        };

        struct SceneLoadCompletion
        {
            SceneHandle handle = {};
            SceneLoadOptions options = {};

            bool success = false;
            std::string error;

            std::unique_ptr<LoadedSceneData> data;
        };

        SceneHandle allocateSceneRecord();

        SceneRecord* record(SceneHandle handle);
        const SceneRecord* record(SceneHandle handle) const;

        void commitSceneLoad(SceneLoadCompletion&& completion);

        static std::unique_ptr<LoadedSceneData> loadSceneFile(
            const std::filesystem::path& path,
            const SceneManagerDesc& desc,
            AssetManager& assets,
            const SceneLoadOptions& options);

        void updateActiveScene(FrameContext& frame);
        void extractRenderView(FrameContext& frame);
        void clearRenderView();

        SceneManagerDesc m_desc = {};
        AssetManager* m_assets = nullptr;
        JobSystem* m_jobs = nullptr;

        bool m_initialized = false;

        mutable std::mutex m_mutex;

        std::vector<std::unique_ptr<SceneRecord>> m_scenes;
        std::vector<uint32_t> m_freeList;

        SceneHandle m_activeScene = {};

        moodycamel::ConcurrentQueue<SceneLoadCompletion> m_completedLoads;

        std::vector<SceneModelRenderItem> m_modelItems;
        std::vector<SceneLightRenderItem> m_lightItems;

        SceneRenderView m_renderView = {};
    };
}
