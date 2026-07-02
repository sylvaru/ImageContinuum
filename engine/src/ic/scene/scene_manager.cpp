#include "ic/common/ic_pch.h"
#include "ic/scene/scene_manager.h"

#include <cassert>
#include <cctype>
#include <cmath>
#include <unordered_map>

#include <spdlog/spdlog.h>
#include <toml++/toml.hpp>

#include "ic/core/app_base.h"
#include "ic/scene/scene_entity.h"
#include "ic/scene/scene_camera_controller_system.h"
#include "ic/scene/scene_transform_system.h"
#include "ic/scene/scene_transform_utils.h"

namespace ic
{
    namespace
    {
        std::string lowerCopy(std::string value)
        {
            std::ranges::transform(
                value,
                value.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return value;
        }

        std::filesystem::path resolveExistingScenePath(
            const std::filesystem::path& path)
        {
            if (path.is_absolute())
            {
                return std::filesystem::weakly_canonical(path);
            }

            std::filesystem::path candidate = path.lexically_normal();
            if (std::filesystem::exists(candidate))
            {
                return std::filesystem::weakly_canonical(candidate);
            }

            std::filesystem::path cursor = std::filesystem::current_path();
            while (!cursor.empty())
            {
                candidate = (cursor / path).lexically_normal();
                if (std::filesystem::exists(candidate))
                {
                    return std::filesystem::weakly_canonical(candidate);
                }

                const std::filesystem::path parent = cursor.parent_path();
                if (parent == cursor)
                {
                    break;
                }

                cursor = parent;
            }

            return path.lexically_normal();
        }

        std::filesystem::path resolveExistingRootedPath(
            const std::filesystem::path& root,
            const std::filesystem::path& path)
        {
            if (root.empty())
            {
                return {};
            }

            std::filesystem::path candidate =
                (root / path).lexically_normal();
            if (std::filesystem::exists(candidate))
            {
                return std::filesystem::weakly_canonical(candidate);
            }

            if (root.is_absolute())
            {
                return candidate;
            }

            std::filesystem::path cursor = std::filesystem::current_path();
            while (!cursor.empty())
            {
                candidate = (cursor / root / path).lexically_normal();
                if (std::filesystem::exists(candidate))
                {
                    return std::filesystem::weakly_canonical(candidate);
                }

                const std::filesystem::path parent = cursor.parent_path();
                if (parent == cursor)
                {
                    break;
                }

                cursor = parent;
            }

            return (root / path).lexically_normal();
        }

        float readFloat(
            const toml::table& table,
            std::string_view key,
            float fallback)
        {
            return table[key].value_or(fallback);
        }

        bool readBool(
            const toml::table& table,
            std::string_view key,
            bool fallback)
        {
            return table[key].value_or(fallback);
        }

        std::string readString(
            const toml::table& table,
            std::string_view key,
            std::string fallback)
        {
            return table[key].value_or(std::move(fallback));
        }

        glm::vec3 readVec3(
            const toml::table& table,
            std::string_view key,
            glm::vec3 fallback)
        {
            const toml::array* array = table[key].as_array();
            if (!array || array->size() < 3)
            {
                return fallback;
            }

            return glm::vec3(
                (*array)[0].value_or(fallback.x),
                (*array)[1].value_or(fallback.y),
                (*array)[2].value_or(fallback.z));
        }

        glm::quat readQuat(
            const toml::table& table,
            std::string_view key,
            glm::quat fallback)
        {
            const toml::array* array = table[key].as_array();
            if (!array || array->size() < 4)
            {
                return fallback;
            }

            const float x = (*array)[0].value_or(fallback.x);
            const float y = (*array)[1].value_or(fallback.y);
            const float z = (*array)[2].value_or(fallback.z);
            const float w = (*array)[3].value_or(fallback.w);
            return glm::normalize(glm::quat(w, x, y, z));
        }

        std::filesystem::path resolveEntityAssetPath(
            const std::filesystem::path& scenePath,
            const std::filesystem::path& modelRoot,
            const std::filesystem::path& assetPath)
        {
            if (assetPath.is_absolute())
            {
                return assetPath.lexically_normal();
            }

            if (!modelRoot.empty())
            {
                const std::filesystem::path rootedPath =
                    resolveExistingRootedPath(modelRoot, assetPath);
                if (std::filesystem::exists(rootedPath))
                {
                    return rootedPath;
                }
            }

            const std::filesystem::path sceneRelativePath =
                (scenePath.parent_path() / assetPath).lexically_normal();
            if (std::filesystem::exists(sceneRelativePath) ||
                modelRoot.empty())
            {
                return sceneRelativePath;
            }

            return (modelRoot / assetPath).lexically_normal();
        }
    }

    SceneManager::~SceneManager()
    {
        shutdown();
    }

    void SceneManager::init(
        SceneManagerDesc desc,
        AssetManager& assets,
        JobSystem& jobs)
    {
        assert(!m_initialized);

        m_desc = desc;
        m_assets = &assets;
        m_jobs = &jobs;
        m_initialized = true;

        spdlog::info("[SceneManager] Initialized");
    }

    void SceneManager::shutdown()
    {
        if (!m_initialized)
        {
            return;
        }

        {
            std::lock_guard lock(m_mutex);
            for (const std::unique_ptr<SceneRecord>& record : m_scenes)
            {
                if (record && record->state == SceneState::Loading)
                {
                    m_jobs->waitForCounter(&record->counter);
                }
            }
        }

        SceneLoadCompletion completion{};
        while (m_completedLoads.try_dequeue(completion))
        {
            commitSceneLoad(std::move(completion));
        }

        {
            std::lock_guard lock(m_mutex);
            m_scenes.clear();
            m_freeList.clear();
            m_activeScene = {};
        }

        clearRenderView();

        m_assets = nullptr;
        m_jobs = nullptr;
        m_initialized = false;

        spdlog::info("[SceneManager] Shutdown complete");
    }

    void SceneManager::update(FrameContext& frame)
    {
        SceneLoadCompletion completion{};
        while (m_completedLoads.try_dequeue(completion))
        {
            commitSceneLoad(std::move(completion));
        }

        updateActiveScene(frame);
        extractRenderView(frame);
    }

    SceneHandle SceneManager::createEmptyScene(std::string name)
    {
        SceneHandle handle = allocateSceneRecord();
        SceneRecord* r = record(handle);
        if (!r)
        {
            return kInvalidSceneHandle;
        }

        r->scene = std::make_unique<Scene>(std::move(name));
        r->state = SceneState::Loaded;

        if (!m_activeScene)
        {
            m_activeScene = handle;
        }

        return handle;
    }

    SceneHandle SceneManager::loadSceneAsync(
        std::filesystem::path path,
        SceneLoadOptions options)
    {
        assert(m_initialized);

        SceneHandle handle = allocateSceneRecord();
        SceneRecord* r = record(handle);
        if (!r)
        {
            return kInvalidSceneHandle;
        }

        r->sourcePath = path;
        r->state = SceneState::Loading;
        r->error.clear();

        if (!m_desc.enableAsyncSceneLoading)
        {
            SceneLoadCompletion completion{};
            completion.handle = handle;
            completion.options = options;

            try
            {
                completion.data =
                    loadSceneFile(path, m_desc.modelRoot, *m_assets, options);
                completion.success = true;
            }
            catch (const std::exception& e)
            {
                completion.error = e.what();
            }

            commitSceneLoad(std::move(completion));
            return handle;
        }

        JobTask task = JobTask::make(
            [this, handle, path = std::move(path), options]()
            {
                SceneLoadCompletion completion{};
                completion.handle = handle;
                completion.options = options;

                try
                {
                    completion.data =
                        loadSceneFile(path, m_desc.modelRoot, *m_assets, options);
                    completion.success = true;
                }
                catch (const std::exception& e)
                {
                    completion.error = e.what();
                }

                m_completedLoads.enqueue(std::move(completion));
            });

        m_jobs->kickTasks(&task, 1, &r->counter);
        return handle;
    }

    void SceneManager::unloadScene(SceneHandle handle)
    {
        std::lock_guard lock(m_mutex);

        SceneRecord* r = record(handle);
        if (!r)
        {
            return;
        }

        if (r->state == SceneState::Loading)
        {
            m_jobs->waitForCounter(&r->counter);
        }

        r->scene.reset();
        r->state = SceneState::Empty;
        r->error.clear();

        if (m_activeScene == handle)
        {
            m_activeScene = {};
            clearRenderView();
        }

        m_freeList.push_back(handle.index);
    }

    void SceneManager::setActiveScene(SceneHandle handle)
    {
        SceneRecord* r = record(handle);
        if (!r || r->state != SceneState::Loaded)
        {
            return;
        }

        m_activeScene = handle;
    }

    Scene* SceneManager::activeScene()
    {
        SceneRecord* r = record(m_activeScene);
        return r ? r->scene.get() : nullptr;
    }

    const Scene* SceneManager::activeScene() const
    {
        const SceneRecord* r = record(m_activeScene);
        return r ? r->scene.get() : nullptr;
    }

    entt::registry* SceneManager::activeRegistry()
    {
        Scene* scene = activeScene();
        return scene ? &scene->registry() : nullptr;
    }

    const entt::registry* SceneManager::activeRegistry() const
    {
        const Scene* scene = activeScene();
        return scene ? &scene->registry() : nullptr;
    }

    SceneState SceneManager::sceneState(SceneHandle handle) const
    {
        const SceneRecord* r = record(handle);
        return r ? r->state : SceneState::Empty;
    }

    std::string SceneManager::sceneError(SceneHandle handle) const
    {
        const SceneRecord* r = record(handle);
        return r ? r->error : std::string{};
    }

    SceneEntity SceneManager::createEntity(std::string_view name)
    {
        Scene* scene = activeScene();
        return scene ? scene->createEntity(name) : SceneEntity{};
    }

    void SceneManager::destroyEntity(SceneEntity entity)
    {
        Scene* scene = activeScene();
        if (!scene || !entity)
        {
            return;
        }

        scene->destroyEntity(entity.handle());
    }

    SceneEntity SceneManager::instantiateModel(
        AssetHandle model,
        std::string_view name)
    {
        SceneEntity entity = createEntity(name);
        if (!entity)
        {
            return {};
        }

        entity.addComponent<ModelComponent>(model, 0u);
        return entity;
    }

    SceneHandle SceneManager::allocateSceneRecord()
    {
        std::lock_guard lock(m_mutex);

        SceneHandle handle{};
        if (!m_freeList.empty())
        {
            handle.index = m_freeList.back();
            m_freeList.pop_back();

            SceneRecord& r = *m_scenes[handle.index];
            handle.generation = r.handle.generation + 1;

            r.handle = handle;
            r.state = SceneState::Empty;
            r.scene.reset();
            r.sourcePath.clear();
            r.error.clear();
        }
        else
        {
            handle.index = static_cast<uint32_t>(m_scenes.size());
            handle.generation = 1;

            auto r = std::make_unique<SceneRecord>();
            r->handle = handle;
            m_scenes.push_back(std::move(r));
        }

        return handle;
    }

    SceneManager::SceneRecord* SceneManager::record(SceneHandle handle)
    {
        return const_cast<SceneRecord*>(
            static_cast<const SceneManager&>(*this).record(handle));
    }

    const SceneManager::SceneRecord* SceneManager::record(SceneHandle handle) const
    {
        if (!handle || handle.index >= m_scenes.size())
        {
            return nullptr;
        }

        const std::unique_ptr<SceneRecord>& r = m_scenes[handle.index];
        if (!r)
        {
            return nullptr;
        }

        return r->handle.generation == handle.generation ? r.get() : nullptr;
    }

    void SceneManager::commitSceneLoad(SceneLoadCompletion&& completion)
    {
        SceneRecord* r = record(completion.handle);
        if (!r)
        {
            return;
        }

        if (!completion.success)
        {
            r->state = SceneState::Failed;
            r->error = std::move(completion.error);

            spdlog::error(
                "[SceneManager] Failed to load scene '{}': {}",
                r->sourcePath.string(),
                r->error);
            return;
        }

        std::unique_ptr<LoadedSceneData> data = std::move(completion.data);
        if (!data)
        {
            r->state = SceneState::Failed;
            r->error = "Scene load completed without scene data";
            return;
        }

        auto scene = std::make_unique<Scene>(data->name);
        scene->setSourcePath(data->sourcePath);

        std::unordered_map<uint64_t, entt::entity> entityByLoadedId;
        entityByLoadedId.reserve(data->entities.size());

        for (const LoadedSceneData::EntityDesc& src : data->entities)
        {
            SceneEntity entity = src.id != 0
                ? scene->createEntityWithId(src.id, src.name)
                : scene->createEntity(src.name);

            entityByLoadedId[entity.getComponent<EntityIdComponent>().id] =
                entity.handle();

            auto& transform = entity.getComponent<TransformComponent>();
            transform.translation = src.translation;
            transform.rotation = src.rotation;
            transform.scale = src.scale;
            transform.dirty = true;

            markTransformDirty(scene->registry(), entity.handle());

            if (src.modelHandle)
            {
                entity.addComponent<ModelComponent>(src.modelHandle, 0u);
            }

            if (src.hasCamera)
            {
                entity.addComponent<CameraComponent>(src.camera);
            }

            if (src.hasFlyCamera)
            {
                entity.addComponent<FlyCameraControllerComponent>(src.flyCamera);
            }

            if (src.hasLight)
            {
                entity.addComponent<LightComponent>(src.light);
            }
        }

        for (const LoadedSceneData::EntityDesc& src : data->entities)
        {
            if (src.parentId < 0 || src.id == 0)
            {
                continue;
            }

            auto childIt = entityByLoadedId.find(src.id);
            auto parentIt =
                entityByLoadedId.find(static_cast<uint64_t>(src.parentId));

            if (childIt == entityByLoadedId.end() ||
                parentIt == entityByLoadedId.end())
            {
                continue;
            }

            setParent(scene->registry(), childIt->second, parentIt->second);
        }

        r->scene = std::move(scene);
        r->state = SceneState::Loaded;
        r->error.clear();

        if (completion.options.makeActive)
        {
            if (completion.options.mode == SceneLoadMode::ReplaceActive ||
                !m_activeScene)
            {
                m_activeScene = completion.handle;
            }
        }

        spdlog::info("[SceneManager] Loaded scene '{}'", r->scene->name());
    }

    std::unique_ptr<SceneManager::LoadedSceneData> SceneManager::loadSceneFile(
        const std::filesystem::path& path,
        const std::filesystem::path& modelRoot,
        AssetManager& assets,
        const SceneLoadOptions& options)
    {
        const std::filesystem::path resolvedPath = resolveExistingScenePath(path);
        toml::table table = toml::parse_file(resolvedPath.string());

        auto result = std::make_unique<LoadedSceneData>();
        result->sourcePath = resolvedPath;

        if (auto* sceneTable = table["scene"].as_table())
        {
            result->name =
                sceneTable->get_as<std::string>("name")
                    ? sceneTable->get_as<std::string>("name")->get()
                    : resolvedPath.stem().string();
        }
        else
        {
            result->name = resolvedPath.stem().string();
        }

        toml::array* entities = table["entities"].as_array();
        if (!entities)
        {
            return result;
        }

        for (const toml::node& node : *entities)
        {
            const toml::table* entityTable = node.as_table();
            if (!entityTable)
            {
                continue;
            }

            LoadedSceneData::EntityDesc desc{};

            if (auto id = entityTable->get_as<int64_t>("id"))
            {
                desc.id = static_cast<uint64_t>(std::max<int64_t>(0, id->get()));
            }

            if (auto name = entityTable->get_as<std::string>("name"))
            {
                desc.name = name->get();
            }

            if (auto model = entityTable->get_as<std::string>("model"))
            {
                desc.modelPath =
                    resolveEntityAssetPath(
                        resolvedPath,
                        modelRoot,
                        model->get());

                if (options.loadReferencedAssets)
                {
                    desc.modelHandle = assets.loadModelAsync(desc.modelPath);
                }
            }

            if (auto parent = entityTable->get_as<int64_t>("parent"))
            {
                desc.parentId = parent->get();
            }

            if (auto* transform = entityTable->get_as<toml::table>("transform"))
            {
                desc.translation =
                    readVec3(*transform, "translation", glm::vec3(0.0f));
                desc.rotation =
                    readQuat(*transform, "rotation", glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
                desc.scale =
                    readVec3(*transform, "scale", glm::vec3(1.0f));
            }

            if (auto* camera = entityTable->get_as<toml::table>("camera"))
            {
                desc.hasCamera = true;
                desc.camera.primary = readBool(*camera, "primary", false);
                desc.camera.verticalFovRadians =
                    glm::radians(readFloat(*camera, "vertical_fov_degrees", 60.0f));
                desc.camera.nearPlane = readFloat(*camera, "near", 0.1f);
                desc.camera.farPlane = readFloat(*camera, "far", 1000.0f);
            }

            if (auto* flyCamera = entityTable->get_as<toml::table>("fly_camera"))
            {
                desc.hasFlyCamera = true;
                desc.flyCamera.moveSpeed =
                    readFloat(*flyCamera, "move_speed", desc.flyCamera.moveSpeed);
                desc.flyCamera.fastMoveMultiplier =
                    readFloat(
                        *flyCamera,
                        "fast_move_multiplier",
                        desc.flyCamera.fastMoveMultiplier);
                desc.flyCamera.mouseSensitivity =
                    readFloat(
                        *flyCamera,
                        "mouse_sensitivity",
                        desc.flyCamera.mouseSensitivity);
                desc.flyCamera.yaw =
                    readFloat(*flyCamera, "yaw", desc.flyCamera.yaw);
                desc.flyCamera.pitch =
                    readFloat(*flyCamera, "pitch", desc.flyCamera.pitch);
                desc.flyCamera.captureMouse =
                    readBool(
                        *flyCamera,
                        "capture_mouse",
                        desc.flyCamera.captureMouse);
            }

            if (auto* light = entityTable->get_as<toml::table>("light"))
            {
                desc.hasLight = true;

                const std::string type =
                    lowerCopy(readString(*light, "type", "point"));
                if (type == "directional")
                {
                    desc.light.type = LightType::Directional;
                }
                else if (type == "spot")
                {
                    desc.light.type = LightType::Spot;
                }
                else
                {
                    desc.light.type = LightType::Point;
                }

                desc.light.color =
                    readVec3(*light, "color", glm::vec3(1.0f));
                desc.light.intensity =
                    readFloat(*light, "intensity", 1.0f);
                desc.light.range =
                    readFloat(*light, "range", 10.0f);
                desc.light.innerConeRadians =
                    glm::radians(readFloat(*light, "inner_cone_degrees", 15.0f));
                desc.light.outerConeRadians =
                    glm::radians(readFloat(*light, "outer_cone_degrees", 30.0f));
            }

            result->entities.push_back(std::move(desc));
        }

        return result;
    }

    void SceneManager::updateActiveScene(FrameContext& frame)
    {
        Scene* scene = activeScene();
        if (!scene)
        {
            return;
        }

        entt::registry& registry = scene->registry();

        if (frame.services && frame.services->input)
        {
            SceneCameraControllerSystem::update(
                registry,
                *frame.services->input,
                frame.deltaTime);
        }

        auto view = registry.view<TransformComponent, VelocityComponent>();

        for (entt::entity entity : view)
        {
            auto& transform = view.get<TransformComponent>(entity);
            const auto& velocity = view.get<VelocityComponent>(entity);

            transform.translation += velocity.linear * frame.deltaTime;
            if (glm::dot(velocity.linear, velocity.linear) > 0.0f)
            {
                markTransformDirty(registry, entity);
            }
        }

        SceneTransformSystem::update(registry);
    }

    void SceneManager::extractRenderView(FrameContext& frame)
    {
        clearRenderView();

        Scene* scene = activeScene();
        if (!scene)
        {
            return;
        }

        entt::registry& registry = scene->registry();

        {
            auto view =
                registry.view<TransformComponent, ModelComponent, ActiveComponent>();
            m_modelItems.reserve(view.size_hint());

            for (entt::entity entity : view)
            {
                const auto& active = view.get<ActiveComponent>(entity);
                if (!active.active)
                {
                    continue;
                }

                const auto& model = view.get<ModelComponent>(entity);
                if (!model.model)
                {
                    continue;
                }

                AssetHandle materialOverride{};
                if (auto* overrideComponent =
                        registry.try_get<MaterialOverrideComponent>(entity))
                {
                    materialOverride = overrideComponent->material;
                }

                const auto& transform = view.get<TransformComponent>(entity);

                SceneModelRenderItem item{};
                item.entity = entity;
                item.model = model.model;
                item.materialOverride = materialOverride;
                item.world = transform.world;
                item.flags = model.flags;
                m_modelItems.push_back(item);
            }
        }

        {
            auto view =
                registry.view<TransformComponent, LightComponent, ActiveComponent>();
            m_lightItems.reserve(view.size_hint());

            for (entt::entity entity : view)
            {
                const auto& active = view.get<ActiveComponent>(entity);
                if (!active.active)
                {
                    continue;
                }

                const auto& transform = view.get<TransformComponent>(entity);
                const auto& light = view.get<LightComponent>(entity);

                SceneLightRenderItem item{};
                item.entity = entity;
                item.type = light.type;
                item.position = glm::vec3(transform.world[3]);
                item.direction = glm::normalize(glm::vec3(
                    transform.world * glm::vec4(0.0f, -1.0f, 0.0f, 0.0f)));
                item.color = light.color;
                item.intensity = light.intensity;
                item.range = light.range;
                item.innerConeRadians = light.innerConeRadians;
                item.outerConeRadians = light.outerConeRadians;
                m_lightItems.push_back(item);
            }
        }

        m_renderView.camera = {};
        m_renderView.models =
            std::span<const SceneModelRenderItem>(m_modelItems.data(), m_modelItems.size());
        m_renderView.lights =
            std::span<const SceneLightRenderItem>(m_lightItems.data(), m_lightItems.size());
        m_renderView.sceneVersion = scene->version();
        m_renderView.frameIndex = frame.frameIndex;

        auto cameraView =
            registry.view<TransformComponent, CameraComponent, ActiveComponent>();
        for (entt::entity entity : cameraView)
        {
            const auto& active = cameraView.get<ActiveComponent>(entity);
            const auto& camera = cameraView.get<CameraComponent>(entity);
            if (!active.active || !camera.primary)
            {
                continue;
            }

            const auto& transform = cameraView.get<TransformComponent>(entity);
            const glm::mat4 viewMatrix = glm::inverse(transform.world);

            const float width = static_cast<float>(frame.windowWidth);
            const float height = static_cast<float>(frame.windowHeight);
            const float aspect = height > 0.0f ? width / height : 1.0f;

            const glm::mat4 projection = glm::perspective(
                camera.verticalFovRadians,
                aspect,
                camera.nearPlane,
                camera.farPlane);

            SceneCameraView extracted{};
            extracted.entity = entity;
            extracted.position = glm::vec3(transform.world[3]);
            extracted.view = viewMatrix;
            extracted.projection = projection;
            extracted.viewProjection = projection * viewMatrix;
            extracted.nearPlane = camera.nearPlane;
            extracted.farPlane = camera.farPlane;
            extracted.verticalFovRadians = camera.verticalFovRadians;
            extracted.aspectRatio = aspect;
            extracted.valid = 1;

            m_renderView.camera = extracted;
            break;
        }
    }

    void SceneManager::clearRenderView()
    {
        m_modelItems.clear();
        m_lightItems.clear();
        m_renderView = {};
    }
}
