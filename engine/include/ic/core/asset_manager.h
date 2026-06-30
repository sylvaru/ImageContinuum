#pragma once

#include "ic/core/job_system.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <concurrentqueue.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace ic
{
    enum class AssetKind : uint8_t
    {
        Unknown = 0,
        Image,
        Model,
        Binary
    };

    enum class AssetState : uint8_t
    {
        Unloaded = 0,
        Queued,
        Loading,
        Loaded,
        Failed
    };

    enum class ImageFormat : uint8_t
    {
        Unknown = 0,
        R8,
        RG8,
        RGB8,
        RGBA8,
        RGBA32F
    };

    struct AssetHandle
    {
        uint32_t index = UINT32_MAX;
        uint32_t generation = 0;

        explicit operator bool() const { return index != UINT32_MAX; }

        friend bool operator==(AssetHandle a, AssetHandle b)
        {
            return a.index == b.index && a.generation == b.generation;
        }

        friend bool operator!=(AssetHandle a, AssetHandle b)
        {
            return !(a == b);
        }
    };

    static constexpr AssetHandle kInvalidAssetHandle{};

    struct AssetHandleHash
    {
        size_t operator()(AssetHandle handle) const noexcept
        {
            const uint64_t packed =
                (static_cast<uint64_t>(handle.generation) << 32ull) |
                static_cast<uint64_t>(handle.index);
            return std::hash<uint64_t>{}(packed);
        }
    };

    struct AssetManagerDesc
    {
        std::filesystem::path assetRoot = "assets";
        uint32_t maxConcurrentLoads = 4;
        bool enableHotReload = false;
    };

    struct ImageLoadOptions
    {
        bool forceRGBA = true;
        bool srgb = true;
        bool flipY = false;
    };

    struct ModelLoadOptions
    {
        bool loadMaterials = true;
        bool loadTextures = true;
        bool generateTangentsIfMissing = false;
        bool mergeMeshes = false;
        bool srgbBaseColorTextures = true;
    };

    struct BinaryLoadOptions
    {
        bool nullTerminate = false;
    };

    struct AssetManifestEntry
    {
        std::string name;
        AssetKind kind = AssetKind::Unknown;
        std::filesystem::path path;
        AssetHandle handle = kInvalidAssetHandle;
    };

    struct LoadedAssetManifest
    {
        std::filesystem::path path;
        std::vector<AssetManifestEntry> assets;
    };

    struct AssetManifestProgress
    {
        uint32_t total = 0;
        uint32_t queued = 0;
        uint32_t loading = 0;
        uint32_t loaded = 0;
        uint32_t failed = 0;

        bool complete() const
        {
            return total != 0 && loaded + failed == total;
        }
    };

    struct AssetLoadError
    {
        std::string message;
    };

    struct ImageAsset
    {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t channels = 0;
        ImageFormat format = ImageFormat::Unknown;
        bool srgb = true;
        std::vector<std::byte> pixels;

        bool valid() const
        {
            return width != 0 &&
                   height != 0 &&
                   !pixels.empty() &&
                   format != ImageFormat::Unknown;
        }
    };

    struct BinaryAsset
    {
        std::vector<std::byte> bytes;
    };

    struct AssetVertex
    {
        glm::vec3 position = {};
        glm::vec3 normal = {};
        glm::vec4 tangent = {};
        glm::vec2 uv0 = {};
        glm::vec2 uv1 = {};
        glm::vec4 color = glm::vec4(1.0f);
    };

    struct AssetBounds
    {
        glm::vec3 min = glm::vec3(0.0f);
        glm::vec3 max = glm::vec3(0.0f);
    };

    struct MeshPrimitiveAsset
    {
        uint32_t firstVertex = 0;
        uint32_t vertexCount = 0;
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;
        uint32_t materialIndex = UINT32_MAX;
        AssetBounds bounds = {};
    };

    struct MeshAsset
    {
        std::string name;
        std::vector<MeshPrimitiveAsset> primitives;
    };

    struct MaterialAsset
    {
        std::string name;
        glm::vec4 baseColorFactor = glm::vec4(1.0f);
        float metallicFactor = 1.0f;
        float roughnessFactor = 1.0f;
        float alphaCutoff = 0.5f;
        int32_t baseColorImage = -1;
        int32_t normalImage = -1;
        int32_t metallicRoughnessImage = -1;
        int32_t emissiveImage = -1;
        bool doubleSided = false;
        bool alphaBlend = false;
        bool alphaMask = false;
    };

    struct NodeAsset
    {
        std::string name;
        int32_t parent = -1;
        int32_t meshIndex = -1;
        glm::vec3 translation = glm::vec3(0.0f);
        glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 scale = glm::vec3(1.0f);
        glm::mat4 localMatrix = glm::mat4(1.0f);
        glm::mat4 worldMatrix = glm::mat4(1.0f);
    };

    struct ModelAsset
    {
        std::string name;
        std::vector<AssetVertex> vertices;
        std::vector<uint32_t> indices;
        std::vector<MeshAsset> meshes;
        std::vector<MaterialAsset> materials;
        std::vector<ImageAsset> images;
        std::vector<NodeAsset> nodes;
        AssetBounds bounds = {};
    };

    using AssetData =
        std::variant<std::monostate, ImageAsset, ModelAsset, BinaryAsset>;

    class AssetManager final
    {
    public:
        AssetManager() = default;
        ~AssetManager();

        AssetManager(const AssetManager&) = delete;
        AssetManager& operator=(const AssetManager&) = delete;

        void init(AssetManagerDesc desc, JobSystem& jobs);
        void shutdown();
        void update();

        AssetHandle loadImageAsync(std::filesystem::path path, ImageLoadOptions options = {});
        AssetHandle loadModelAsync(std::filesystem::path path, ModelLoadOptions options = {});
        AssetHandle loadBinaryAsync(std::filesystem::path path, BinaryLoadOptions options = {});
        AssetHandle loadAsync(std::filesystem::path path);
        LoadedAssetManifest loadManifestAsync(std::filesystem::path path);
        [[nodiscard]] AssetManifestProgress manifestProgress(
            const LoadedAssetManifest& manifest) const;

        void unload(AssetHandle handle);
        void wait(AssetHandle handle);

        [[nodiscard]] AssetState state(AssetHandle handle) const;
        [[nodiscard]] bool isLoaded(AssetHandle handle) const;
        [[nodiscard]] bool failed(AssetHandle handle) const;
        [[nodiscard]] std::string error(AssetHandle handle) const;
        [[nodiscard]] AssetKind kind(AssetHandle handle) const;
        [[nodiscard]] std::filesystem::path path(AssetHandle handle) const;
        [[nodiscard]] const ImageAsset* image(AssetHandle handle) const;
        [[nodiscard]] const ModelAsset* model(AssetHandle handle) const;
        [[nodiscard]] const BinaryAsset* binary(AssetHandle handle) const;
        [[nodiscard]] const AssetData* data(AssetHandle handle) const;

    private:
        struct AssetKey
        {
            AssetKind kind = AssetKind::Unknown;
            std::string canonicalPath;
            uint64_t optionHash = 0;

            friend bool operator==(const AssetKey& a, const AssetKey& b)
            {
                return a.kind == b.kind &&
                       a.canonicalPath == b.canonicalPath &&
                       a.optionHash == b.optionHash;
            }
        };

        struct AssetKeyHash
        {
            size_t operator()(const AssetKey& key) const noexcept;
        };

        struct AssetRecord
        {
            AssetHandle handle = {};
            AssetKind kind = AssetKind::Unknown;
            std::filesystem::path path;
            AssetKey key;
            std::atomic<AssetState> state = AssetState::Unloaded;
            mutable std::shared_mutex mutex;
            std::shared_ptr<const AssetData> data;
            std::string error;
            JobCounter counter;
            uint32_t generation = 1;
        };

        struct AssetCompletion
        {
            AssetHandle handle = {};
            AssetKind kind = AssetKind::Unknown;
            std::shared_ptr<AssetData> data;
            std::string error;
            bool success = false;
        };

        AssetHandle requestLoad(AssetKind kind, std::filesystem::path path, uint64_t optionHash);
        void kickImageLoad(AssetHandle handle, std::filesystem::path path, ImageLoadOptions options);
        void kickModelLoad(AssetHandle handle, std::filesystem::path path, ModelLoadOptions options);
        void kickBinaryLoad(AssetHandle handle, std::filesystem::path path, BinaryLoadOptions options);
        void commitCompletion(AssetCompletion&& completion);

        [[nodiscard]] AssetRecord* record(AssetHandle handle);
        [[nodiscard]] const AssetRecord* record(AssetHandle handle) const;
        [[nodiscard]] std::filesystem::path resolvePath(const std::filesystem::path& path) const;
        [[nodiscard]] std::filesystem::path resolveExistingPath(const std::filesystem::path& path) const;
        [[nodiscard]] std::string canonicalKeyPath(const std::filesystem::path& path) const;

        [[nodiscard]] static AssetKind detectKind(const std::filesystem::path& path);
        [[nodiscard]] static uint64_t hashOptions(const ImageLoadOptions& options);
        [[nodiscard]] static uint64_t hashOptions(const ModelLoadOptions& options);
        [[nodiscard]] static uint64_t hashOptions(const BinaryLoadOptions& options);
        [[nodiscard]] static std::vector<std::byte> readWholeFile(const std::filesystem::path& path, bool nullTerminate);
        [[nodiscard]] static ImageAsset decodeImage(std::span<const std::byte> bytes, const ImageLoadOptions& options);
        [[nodiscard]] static ModelAsset decodeModel(const std::filesystem::path& path, const ModelLoadOptions& options);

        AssetManagerDesc m_desc = {};
        JobSystem* m_jobs = nullptr;
        bool m_initialized = false;
        mutable std::shared_mutex m_recordsMutex;
        std::vector<std::unique_ptr<AssetRecord>> m_records;
        std::vector<uint32_t> m_freeList;
        std::unordered_map<AssetKey, AssetHandle, AssetKeyHash> m_cache;
        moodycamel::ConcurrentQueue<AssetCompletion> m_completed;
    };
}
