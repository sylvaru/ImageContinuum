#include "ic/core/asset_manager.h"
#include "ic/core/task.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <spdlog/spdlog.h>
#include <toml++/toml.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace ic
{
    namespace
    {
        void hashCombine(uint64_t& seed, uint64_t value)
        {
            seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6ull) + (seed >> 2ull);
        }

        std::string lowerCopy(std::string value)
        {
            std::ranges::transform(
                value,
                value.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return value;
        }

        glm::mat4 toGlm(const fastgltf::math::fmat4x4& m)
        {
            glm::mat4 out(1.0f);
            for (size_t c = 0; c < 4; ++c)
            {
                for (size_t r = 0; r < 4; ++r)
                {
                    out[static_cast<int>(c)][static_cast<int>(r)] = m[c][r];
                }
            }
            return out;
        }

        AssetBounds boundsFromPositions(const std::vector<glm::vec3>& positions)
        {
            AssetBounds bounds{};
            if (positions.empty())
            {
                return bounds;
            }

            bounds.min = positions.front();
            bounds.max = positions.front();

            for (const glm::vec3& p : positions)
            {
                bounds.min = glm::min(bounds.min, p);
                bounds.max = glm::max(bounds.max, p);
            }

            return bounds;
        }

        void expandBounds(AssetBounds& dst, const AssetBounds& src, bool& initialized)
        {
            if (!initialized)
            {
                dst = src;
                initialized = true;
                return;
            }

            dst.min = glm::min(dst.min, src.min);
            dst.max = glm::max(dst.max, src.max);
        }

        AssetKind assetKindFromString(std::string_view value)
        {
            if (value == "image")
            {
                return AssetKind::Image;
            }

            if (value == "model")
            {
                return AssetKind::Model;
            }

            if (value == "binary")
            {
                return AssetKind::Binary;
            }

            return AssetKind::Unknown;
        }

        bool tomlBoolOr(const toml::table& table, std::string_view key, bool fallback)
        {
            return table[key].value_or(fallback);
        }

        std::vector<std::byte> copyBytes(std::span<const std::byte> bytes)
        {
            return std::vector<std::byte>(bytes.begin(), bytes.end());
        }

        TextureWrapMode convertWrap(fastgltf::Wrap wrap)
        {
            switch (wrap)
            {
            case fastgltf::Wrap::Repeat:
                return TextureWrapMode::Repeat;
            case fastgltf::Wrap::MirroredRepeat:
                return TextureWrapMode::MirroredRepeat;
            case fastgltf::Wrap::ClampToEdge:
                return TextureWrapMode::ClampToEdge;
            }

            return TextureWrapMode::Repeat;
        }

        TextureFilterMode convertFilter(fastgltf::Optional<fastgltf::Filter> filter)
        {
            if (!filter.has_value())
            {
                return TextureFilterMode::Default;
            }

            switch (*filter)
            {
            case fastgltf::Filter::Nearest:
                return TextureFilterMode::Nearest;
            case fastgltf::Filter::Linear:
                return TextureFilterMode::Linear;
            case fastgltf::Filter::NearestMipMapNearest:
                return TextureFilterMode::NearestMipmapNearest;
            case fastgltf::Filter::LinearMipMapNearest:
                return TextureFilterMode::LinearMipmapNearest;
            case fastgltf::Filter::NearestMipMapLinear:
                return TextureFilterMode::NearestMipmapLinear;
            case fastgltf::Filter::LinearMipMapLinear:
                return TextureFilterMode::LinearMipmapLinear;
            }

            return TextureFilterMode::Default;
        }

        SamplerAsset decodeSampler(const fastgltf::Sampler& src)
        {
            SamplerAsset dst{};
            dst.name = std::string(src.name);
            dst.minFilter = convertFilter(src.minFilter);
            dst.magFilter = convertFilter(src.magFilter);
            dst.wrapU = convertWrap(src.wrapS);
            dst.wrapV = convertWrap(src.wrapT);
            return dst;
        }

        MaterialTextureSlot decodeTextureInfo(
            const fastgltf::TextureInfo& info,
            TextureTransferFunction transfer)
        {
            MaterialTextureSlot slot{};
            slot.textureIndex = static_cast<int32_t>(info.textureIndex);
            slot.texCoord = static_cast<uint32_t>(info.texCoordIndex);
            slot.transfer = transfer;
            return slot;
        }

        MaterialTextureSlot decodeNormalTextureInfo(
            const fastgltf::NormalTextureInfo& info)
        {
            MaterialTextureSlot slot =
                decodeTextureInfo(info, TextureTransferFunction::Linear);
            slot.scale = static_cast<float>(info.scale);
            return slot;
        }

        MaterialTextureSlot decodeOcclusionTextureInfo(
            const fastgltf::OcclusionTextureInfo& info)
        {
            MaterialTextureSlot slot =
                decodeTextureInfo(info, TextureTransferFunction::Linear);
            slot.strength = static_cast<float>(info.strength);
            return slot;
        }

        std::vector<std::byte> readWholeFileBytes(
            const std::filesystem::path& path,
            bool nullTerminate)
        {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file)
            {
                throw std::runtime_error("Could not open file: " + path.string());
            }

            const std::streamsize size = file.tellg();
            if (size < 0)
            {
                throw std::runtime_error("Could not determine file size: " + path.string());
            }

            file.seekg(0, std::ios::beg);

            std::vector<std::byte> bytes(
                static_cast<size_t>(size) + (nullTerminate ? 1u : 0u));

            if (size > 0)
            {
                file.read(reinterpret_cast<char*>(bytes.data()), size);
                if (!file)
                {
                    throw std::runtime_error("Could not read file: " + path.string());
                }
            }

            if (nullTerminate)
            {
                bytes[static_cast<size_t>(size)] = std::byte{ 0 };
            }

            return bytes;
        }

        ImageAsset decodeImageBytes(
            std::span<const std::byte> bytes,
            const ImageLoadOptions& options)
        {
            if (bytes.empty())
            {
                throw std::runtime_error("Image byte span is empty");
            }

            stbi_set_flip_vertically_on_load_thread(options.flipY ? 1 : 0);

            int width = 0;
            int height = 0;
            int sourceChannels = 0;
            const int desiredChannels = options.forceRGBA ? 4 : 0;

            if (stbi_is_hdr_from_memory(
                    reinterpret_cast<const stbi_uc*>(bytes.data()),
                    static_cast<int>(bytes.size())) != 0)
            {
                float* decoded = stbi_loadf_from_memory(
                    reinterpret_cast<const stbi_uc*>(bytes.data()),
                    static_cast<int>(bytes.size()),
                    &width,
                    &height,
                    &sourceChannels,
                    desiredChannels);

                if (!decoded)
                {
                    const char* reason = stbi_failure_reason();
                    throw std::runtime_error(
                        std::string("stbi_loadf_from_memory failed: ") +
                        (reason ? reason : "unknown error"));
                }

                const uint32_t outputChannels =
                    options.forceRGBA ? 4u : static_cast<uint32_t>(sourceChannels);
                if (outputChannels != 4)
                {
                    stbi_image_free(decoded);
                    throw std::runtime_error("HDR images must decode to RGBA32F");
                }

                const size_t pixelByteCount =
                    static_cast<size_t>(width) *
                    static_cast<size_t>(height) *
                    static_cast<size_t>(outputChannels) *
                    sizeof(float);

                ImageAsset image{};
                image.width = static_cast<uint32_t>(width);
                image.height = static_cast<uint32_t>(height);
                image.channels = outputChannels;
                image.format = ImageFormat::RGBA32F;
                image.srgb = false;
                image.pixels.resize(pixelByteCount);
                std::memcpy(image.pixels.data(), decoded, pixelByteCount);
                stbi_image_free(decoded);
                return image;
            }

            stbi_uc* decoded = stbi_load_from_memory(
                reinterpret_cast<const stbi_uc*>(bytes.data()),
                static_cast<int>(bytes.size()),
                &width,
                &height,
                &sourceChannels,
                desiredChannels);

            if (!decoded)
            {
                const char* reason = stbi_failure_reason();
                throw std::runtime_error(
                    std::string("stbi_load_from_memory failed: ") +
                    (reason ? reason : "unknown error"));
            }

            const uint32_t outputChannels =
                options.forceRGBA ? 4u : static_cast<uint32_t>(sourceChannels);
            const size_t pixelByteCount =
                static_cast<size_t>(width) *
                static_cast<size_t>(height) *
                static_cast<size_t>(outputChannels);

            ImageAsset image{};
            image.width = static_cast<uint32_t>(width);
            image.height = static_cast<uint32_t>(height);
            image.channels = outputChannels;
            image.srgb = options.srgb;

            switch (outputChannels)
            {
            case 1:
                image.format = ImageFormat::R8;
                break;
            case 2:
                image.format = ImageFormat::RG8;
                break;
            case 3:
                image.format = ImageFormat::RGB8;
                break;
            case 4:
                image.format = ImageFormat::RGBA8;
                break;
            default:
                stbi_image_free(decoded);
                throw std::runtime_error("Unsupported image channel count");
            }

            image.pixels.resize(pixelByteCount);
            std::memcpy(image.pixels.data(), decoded, pixelByteCount);
            stbi_image_free(decoded);
            return image;
        }

        std::vector<std::byte> extractGltfImageBytes(
            const fastgltf::Asset& gltf,
            const fastgltf::Image& image,
            const std::filesystem::path& modelDirectory)
        {
            return std::visit(
                fastgltf::visitor{
                    [](const fastgltf::sources::Array& array)
                    {
                        return copyBytes(
                            std::span<const std::byte>(
                                array.bytes.data(),
                                array.bytes.size_bytes()));
                    },
                    [](const fastgltf::sources::Vector& vector)
                    {
                        return copyBytes(vector.bytes);
                    },
                    [](const fastgltf::sources::ByteView& view)
                    {
                        return copyBytes(view.bytes);
                    },
                    [&](const fastgltf::sources::BufferView& view)
                    {
                        if (view.bufferViewIndex >= gltf.bufferViews.size())
                        {
                            throw std::runtime_error("glTF image references an invalid buffer view");
                        }

                        fastgltf::DefaultBufferDataAdapter adapter{};
                        return copyBytes(adapter(gltf, view.bufferViewIndex));
                    },
                    [&](const fastgltf::sources::URI& uri)
                    {
                        if (!uri.uri.isLocalPath())
                        {
                            throw std::runtime_error("glTF image URI is not a local path");
                        }

                        const std::string uriPath(
                            uri.uri.path().begin(),
                            uri.uri.path().end());
                        const std::filesystem::path imagePath =
                            (modelDirectory / uriPath).lexically_normal();
                        std::vector<std::byte> bytes =
                            readWholeFileBytes(imagePath, false);

                        if (uri.fileByteOffset > bytes.size())
                        {
                            throw std::runtime_error("glTF image URI byte offset is out of range");
                        }

                        if (uri.fileByteOffset == 0)
                        {
                            return bytes;
                        }

                        return std::vector<std::byte>(
                            bytes.begin() + static_cast<ptrdiff_t>(uri.fileByteOffset),
                            bytes.end());
                    },
                    [](const auto&) -> std::vector<std::byte>
                    {
                        throw std::runtime_error("Unsupported glTF image data source");
                    }},
                image.data);
        }

        ImageAsset decodeGltfImage(
            const std::vector<std::byte>& bytes,
            bool flipY,
            std::string_view debugName)
        {
            ImageLoadOptions options{};
            options.forceRGBA = true;
            options.srgb = false;
            options.flipY = flipY;

            try
            {
                return decodeImageBytes(bytes, options);
            }
            catch (const std::exception& e)
            {
                throw std::runtime_error(
                    "Failed to decode glTF image '" +
                    std::string(debugName) +
                    "': " +
                    e.what());
            }
        }

        glm::vec3 toGlm(const fastgltf::math::fvec3& v)
        {
            return glm::vec3(v[0], v[1], v[2]);
        }

        glm::quat toGlm(const fastgltf::math::fquat& q)
        {
            return glm::quat(q[3], q[0], q[1], q[2]);
        }
    }

    size_t AssetManager::AssetKeyHash::operator()(const AssetKey& key) const noexcept
    {
        size_t h = std::hash<std::string>{}(key.canonicalPath);
        h ^= std::hash<uint64_t>{}(key.optionHash) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(static_cast<uint32_t>(key.kind)) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }

    AssetManager::~AssetManager()
    {
        shutdown();
    }

    void AssetManager::init(AssetManagerDesc desc, JobSystem& jobs)
    {
        assert(!m_initialized);

        m_desc = std::move(desc);
        m_desc.maxConcurrentLoads = std::max(1u, m_desc.maxConcurrentLoads);
        m_jobs = &jobs;
        m_initialized = true;

        spdlog::info(
            "[AssetManager] Initialized. root='{}', maxConcurrentLoads={}",
            m_desc.assetRoot.string(),
            m_desc.maxConcurrentLoads);
    }

    void AssetManager::shutdown()
    {
        if (!m_initialized)
        {
            return;
        }

        spdlog::info("[AssetManager] Shutting down...");

        {
            std::shared_lock lock(m_recordsMutex);
            for (auto& r : m_records)
            {
                if (r &&
                    r->state.load(std::memory_order_acquire) == AssetState::Loading)
                {
                    m_jobs->waitForCounter(&r->counter);
                }
            }
        }

        update();

        {
            std::unique_lock lock(m_recordsMutex);
            m_cache.clear();
            m_freeList.clear();
            m_records.clear();
        }

        AssetCompletion completion{};
        while (m_completed.try_dequeue(completion))
        {
        }

        m_jobs = nullptr;
        m_initialized = false;

        spdlog::info("[AssetManager] Shutdown complete");
    }

    void AssetManager::update()
    {
        AssetCompletion completion{};
        while (m_completed.try_dequeue(completion))
        {
            commitCompletion(std::move(completion));
        }
    }

    AssetHandle AssetManager::loadImageAsync(std::filesystem::path path, ImageLoadOptions options)
    {
        AssetHandle handle = requestLoad(AssetKind::Image, std::move(path), hashOptions(options));
        AssetRecord* r = record(handle);
        if (!r)
        {
            return kInvalidAssetHandle;
        }

        AssetState expected = AssetState::Queued;
        if (r->state.compare_exchange_strong(expected, AssetState::Loading, std::memory_order_acq_rel))
        {
            kickImageLoad(handle, r->path, options);
        }

        return handle;
    }

    AssetHandle AssetManager::loadModelAsync(std::filesystem::path path, ModelLoadOptions options)
    {
        AssetHandle handle = requestLoad(AssetKind::Model, std::move(path), hashOptions(options));
        AssetRecord* r = record(handle);
        if (!r)
        {
            return kInvalidAssetHandle;
        }

        AssetState expected = AssetState::Queued;
        if (r->state.compare_exchange_strong(expected, AssetState::Loading, std::memory_order_acq_rel))
        {
            kickModelLoad(handle, r->path, options);
        }

        return handle;
    }

    AssetHandle AssetManager::loadBinaryAsync(std::filesystem::path path, BinaryLoadOptions options)
    {
        AssetHandle handle = requestLoad(AssetKind::Binary, std::move(path), hashOptions(options));
        AssetRecord* r = record(handle);
        if (!r)
        {
            return kInvalidAssetHandle;
        }

        AssetState expected = AssetState::Queued;
        if (r->state.compare_exchange_strong(expected, AssetState::Loading, std::memory_order_acq_rel))
        {
            kickBinaryLoad(handle, r->path, options);
        }

        return handle;
    }

    AssetHandle AssetManager::loadAsync(std::filesystem::path path)
    {
        switch (detectKind(path))
        {
        case AssetKind::Image:
            return loadImageAsync(std::move(path));
        case AssetKind::Model:
            return loadModelAsync(std::move(path));
        case AssetKind::Binary:
            return loadBinaryAsync(std::move(path));
        default:
            spdlog::warn("[AssetManager] Unknown asset type: {}", path.string());
            return kInvalidAssetHandle;
        }
    }

    LoadedAssetManifest AssetManager::loadManifestAsync(std::filesystem::path path)
    {
        LoadedAssetManifest manifest{};
        manifest.path = resolveExistingPath(path);

        if (!std::filesystem::exists(manifest.path))
        {
            spdlog::error(
                "[AssetManager] Asset manifest not found: {}",
                manifest.path.string());
            return manifest;
        }

        toml::table root{};

        try
        {
            root = toml::parse_file(manifest.path.string());
        }
        catch (const toml::parse_error& e)
        {
            spdlog::error(
                "[AssetManager] Could not parse asset manifest '{}': {}",
                manifest.path.string(),
                e.description());
            return manifest;
        }

        toml::array* assets = root["assets"].as_array();
        if (!assets)
        {
            spdlog::warn(
                "[AssetManager] Asset manifest has no [[assets]] entries: {}",
                manifest.path.string());
            return manifest;
        }

        const std::filesystem::path baseDir = manifest.path.parent_path();

        assets->for_each(
            [&](toml::table& assetTable)
            {
                const std::string name =
                    assetTable["name"].value_or<std::string>({});
                const std::string type =
                    lowerCopy(assetTable["type"].value_or<std::string>({}));
                const std::string pathString =
                    assetTable["path"].value_or<std::string>({});

                if (name.empty() || pathString.empty())
                {
                    spdlog::warn(
                        "[AssetManager] Skipping manifest asset with missing name/path in {}",
                        manifest.path.string());
                    return;
                }

                std::filesystem::path assetPath(pathString);
                if (assetPath.is_relative())
                {
                    assetPath = baseDir / assetPath;
                }
                assetPath = assetPath.lexically_normal();

                AssetKind kind = assetKindFromString(type);
                if (kind == AssetKind::Unknown)
                {
                    kind = detectKind(assetPath);
                }

                AssetHandle handle = kInvalidAssetHandle;

                switch (kind)
                {
                case AssetKind::Image:
                {
                    ImageLoadOptions options{};
                    options.forceRGBA = tomlBoolOr(assetTable, "force_rgba", options.forceRGBA);
                    options.srgb = tomlBoolOr(assetTable, "srgb", options.srgb);
                    options.flipY = tomlBoolOr(assetTable, "flip_y", options.flipY);
                    handle = loadImageAsync(assetPath, options);
                    break;
                }

                case AssetKind::Model:
                {
                    ModelLoadOptions options{};
                    options.loadMaterials = tomlBoolOr(assetTable, "load_materials", options.loadMaterials);
                    options.loadTextures = tomlBoolOr(assetTable, "load_textures", options.loadTextures);
                    options.generateTangentsIfMissing = tomlBoolOr(assetTable, "generate_tangents_if_missing", options.generateTangentsIfMissing);
                    options.mergeMeshes = tomlBoolOr(assetTable, "merge_meshes", options.mergeMeshes);
                    handle = loadModelAsync(assetPath, options);
                    break;
                }

                case AssetKind::Binary:
                {
                    BinaryLoadOptions options{};
                    options.nullTerminate = tomlBoolOr(assetTable, "null_terminate", options.nullTerminate);
                    handle = loadBinaryAsync(assetPath, options);
                    break;
                }

                default:
                    spdlog::warn(
                        "[AssetManager] Skipping unknown manifest asset type '{}' for '{}'",
                        type,
                        name);
                    break;
                }

                if (handle)
                {
                    manifest.assets.push_back(
                        AssetManifestEntry{
                            .name = name,
                            .kind = kind,
                            .path = assetPath,
                            .handle = handle,
                        });
                }
            });

        spdlog::info(
            "[AssetManager] Queued {} assets from manifest '{}'",
            manifest.assets.size(),
            manifest.path.string());

        return manifest;
    }

    AssetManifestProgress AssetManager::manifestProgress(
        const LoadedAssetManifest& manifest) const
    {
        AssetManifestProgress progress{};
        progress.total = static_cast<uint32_t>(manifest.assets.size());

        for (const AssetManifestEntry& entry : manifest.assets)
        {
            switch (state(entry.handle))
            {
            case AssetState::Queued:
                ++progress.queued;
                break;

            case AssetState::Loading:
                ++progress.loading;
                break;

            case AssetState::Loaded:
                ++progress.loaded;
                break;

            case AssetState::Failed:
                ++progress.failed;
                break;

            default:
                break;
            }
        }

        return progress;
    }

    void AssetManager::unload(AssetHandle handle)
    {
        AssetRecord* r = record(handle);
        if (!r)
        {
            return;
        }

        if (r->state.load(std::memory_order_acquire) == AssetState::Loading)
        {
            sync_wait(
                *m_jobs,
                waitForCounterAsync(*m_jobs, r->counter));
            update();
        }

        {
            std::unique_lock lock(r->mutex);
            r->data.reset();
            r->error.clear();
        }

        {
            std::unique_lock recordsLock(m_recordsMutex);
            m_cache.erase(r->key);
        }

        r->state.store(AssetState::Unloaded, std::memory_order_release);
    }

    void AssetManager::wait(AssetHandle handle)
    {
        AssetRecord* r = record(handle);
        if (!r)
        {
            return;
        }

        if (r->state.load(std::memory_order_acquire) == AssetState::Loading)
        {
            // Coroutine form of the wait. Where the old path spun on the counter
            // with _mm_pause (waitForCounter), we now express the wait as a
            // coroutine that SUSPENDS on the counter and is resumed via the job
            // queue the instant the counter hits zero. sync_wait drives it from
            // this synchronous call site, pumping the queue so the calling thread
            // still helps drain work and cannot deadlock. Behaviour is identical
            // to before (block until loaded, then flush completions); the wait is
            // just no longer a busy spin.
            sync_wait(
                *m_jobs,
                waitForCounterAsync(*m_jobs, r->counter));
            update();
        }
    }

    AssetState AssetManager::state(AssetHandle handle) const
    {
        const AssetRecord* r = record(handle);
        return r ? r->state.load(std::memory_order_acquire) : AssetState::Unloaded;
    }

    bool AssetManager::isLoaded(AssetHandle handle) const
    {
        return state(handle) == AssetState::Loaded;
    }

    bool AssetManager::failed(AssetHandle handle) const
    {
        return state(handle) == AssetState::Failed;
    }

    std::string AssetManager::error(AssetHandle handle) const
    {
        const AssetRecord* r = record(handle);
        if (!r)
        {
            return {};
        }

        std::shared_lock lock(r->mutex);
        return r->error;
    }

    AssetKind AssetManager::kind(AssetHandle handle) const
    {
        const AssetRecord* r = record(handle);
        return r ? r->kind : AssetKind::Unknown;
    }

    std::filesystem::path AssetManager::path(AssetHandle handle) const
    {
        const AssetRecord* r = record(handle);
        return r ? r->path : std::filesystem::path{};
    }

    const ImageAsset* AssetManager::image(AssetHandle handle) const
    {
        const AssetData* asset = data(handle);
        return asset ? std::get_if<ImageAsset>(asset) : nullptr;
    }

    const ModelAsset* AssetManager::model(AssetHandle handle) const
    {
        const AssetData* asset = data(handle);
        return asset ? std::get_if<ModelAsset>(asset) : nullptr;
    }

    const BinaryAsset* AssetManager::binary(AssetHandle handle) const
    {
        const AssetData* asset = data(handle);
        return asset ? std::get_if<BinaryAsset>(asset) : nullptr;
    }

    const AssetData* AssetManager::data(AssetHandle handle) const
    {
        std::shared_ptr<const AssetData> asset = dataShared(handle);
        return asset ? asset.get() : nullptr;
    }

    std::shared_ptr<const ImageAsset> AssetManager::imageShared(AssetHandle handle) const
    {
        std::shared_ptr<const AssetData> asset = dataShared(handle);
        if (!asset)
        {
            return nullptr;
        }

        const ImageAsset* image = std::get_if<ImageAsset>(asset.get());
        return image ? std::shared_ptr<const ImageAsset>(std::move(asset), image) : nullptr;
    }

    std::shared_ptr<const ModelAsset> AssetManager::modelShared(AssetHandle handle) const
    {
        std::shared_ptr<const AssetData> asset = dataShared(handle);
        if (!asset)
        {
            return nullptr;
        }

        const ModelAsset* model = std::get_if<ModelAsset>(asset.get());
        return model ? std::shared_ptr<const ModelAsset>(std::move(asset), model) : nullptr;
    }

    std::shared_ptr<const BinaryAsset> AssetManager::binaryShared(AssetHandle handle) const
    {
        std::shared_ptr<const AssetData> asset = dataShared(handle);
        if (!asset)
        {
            return nullptr;
        }

        const BinaryAsset* binary = std::get_if<BinaryAsset>(asset.get());
        return binary ? std::shared_ptr<const BinaryAsset>(std::move(asset), binary) : nullptr;
    }

    std::shared_ptr<const AssetData> AssetManager::dataShared(AssetHandle handle) const
    {
        const AssetRecord* r = record(handle);
        if (!r)
        {
            return nullptr;
        }

        std::shared_lock lock(r->mutex);
        return r->data;
    }

    AssetHandle AssetManager::requestLoad(
        AssetKind kind,
        std::filesystem::path path,
        uint64_t optionHash)
    {
        assert(m_initialized);

        const std::filesystem::path resolved = resolvePath(path);
        AssetKey key{ kind, canonicalKeyPath(resolved), optionHash };

        std::unique_lock lock(m_recordsMutex);

        if (auto it = m_cache.find(key); it != m_cache.end())
        {
            return it->second;
        }

        uint32_t index = 0;
        AssetRecord* r = nullptr;

        if (!m_freeList.empty())
        {
            index = m_freeList.back();
            m_freeList.pop_back();
            m_records[index] = std::make_unique<AssetRecord>();
            r = m_records[index].get();
        }
        else
        {
            index = static_cast<uint32_t>(m_records.size());
            m_records.push_back(std::make_unique<AssetRecord>());
            r = m_records.back().get();
        }

        r->generation = std::max(1u, r->generation);
        r->handle = AssetHandle{ index, r->generation };
        r->kind = kind;
        r->path = resolved;
        r->key = key;
        r->state.store(AssetState::Queued, std::memory_order_release);

        m_cache.emplace(std::move(key), r->handle);
        return r->handle;
    }

    void AssetManager::kickImageLoad(
        AssetHandle handle,
        std::filesystem::path path,
        ImageLoadOptions options)
    {
        JobTask task = JobTask::make([this, handle, path = std::move(path), options]()
        {
            AssetCompletion completion{};
            completion.handle = handle;
            completion.kind = AssetKind::Image;

            try
            {
                std::vector<std::byte> bytes = readWholeFile(path, false);
                completion.data = std::make_shared<AssetData>(decodeImage(bytes, options));
                completion.success = true;
            }
            catch (const std::exception& e)
            {
                completion.error = e.what();
            }

            m_completed.enqueue(std::move(completion));
        });

        AssetRecord* r = record(handle);
        if (r)
        {
            m_jobs->kickTasks(&task, 1, &r->counter);
        }
    }

    void AssetManager::kickModelLoad(
        AssetHandle handle,
        std::filesystem::path path,
        ModelLoadOptions options)
    {

        spdlog::info(
            "[AssetManager] Queueing model load handle={}:{} path='{}'",
            handle.index,
            handle.generation,
            path.string());

        JobTask task = JobTask::make([this, handle, path = std::move(path), options]()
        {
            AssetCompletion completion{};
            completion.handle = handle;
            completion.kind = AssetKind::Model;


            spdlog::info(
                "[AssetManager] Model load job started handle={}:{} path='{}'",
                handle.index,
                handle.generation,
                path.string());

            try
            {
                spdlog::info(
                    "[AssetManager] Decoding model handle={}:{} path='{}'",
                    handle.index,
                    handle.generation,
                    path.string());

                ModelAsset model = decodeModel(path, options);

                spdlog::info(
                    "[AssetManager] Decoded model handle={}:{} path='{}': "
                    "{} vertices, {} indices, {} meshes, {} materials, "
                    "{} images, {} textures, {} samplers, {} nodes, {} scenes",
                    handle.index,
                    handle.generation,
                    path.string(),
                    model.vertices.size(),
                    model.indices.size(),
                    model.meshes.size(),
                    model.materials.size(),
                    model.images.size(),
                    model.textures.size(),
                    model.samplers.size(),
                    model.nodes.size(),
                    model.scenes.size());

                completion.data = std::make_shared<AssetData>(std::move(model));
                completion.success = true;
            }
            catch (const std::exception& e)
            {
                completion.error = e.what();
                completion.success = false;
                spdlog::error(
                    "[AssetManager] Model load failed handle={}:{} path='{}': {}",
                    handle.index,
                    handle.generation,
                    path.string(),
                    e.what());
            }

            m_completed.enqueue(std::move(completion));
        });

        AssetRecord* r = record(handle);
        if (r)
        {
            m_jobs->kickTasks(&task, 1, &r->counter);
        }
        else
        {
            spdlog::error(
                "[AssetManager] Could not kick model load; missing record handle={}:{} path='{}'",
                handle.index,
                handle.generation,
                path.string());
        }
    }

    void AssetManager::kickBinaryLoad(
        AssetHandle handle,
        std::filesystem::path path,
        BinaryLoadOptions options)
    {
        spdlog::info(
            "[AssetManager] Queueing binary load handle={}:{} path='{}'",
            handle.index,
            handle.generation,
            path.string());


        JobTask task = JobTask::make([this, handle, path = std::move(path), options]()
        {
            AssetCompletion completion{};
            completion.handle = handle;
            completion.kind = AssetKind::Binary;

            spdlog::info(
                "[AssetManager] Binary load job started handle={}:{} path='{}'",
                handle.index,
                handle.generation,
                path.string());

            try
            {
                BinaryAsset binary{};
                binary.bytes = readWholeFile(path, options.nullTerminate);

                spdlog::info(
                    "[AssetManager] Binary load succeeded handle={}:{} path='{}' bytes={}",
                    handle.index,
                    handle.generation,
                    path.string(),
                    binary.bytes.size());

                completion.data = std::make_shared<AssetData>(std::move(binary));
                completion.success = true;
            }
            catch (const std::exception& e)
            {
                completion.success = false;
                completion.error = e.what();

                spdlog::error(
                    "[AssetManager] Binary load failed handle={}:{} path='{}': {}",
                    handle.index,
                    handle.generation,
                    path.string(),
                    e.what());
            }

            m_completed.enqueue(std::move(completion));
        });

        AssetRecord* r = record(handle);
        if (r)
        {
            m_jobs->kickTasks(&task, 1, &r->counter);
        }
        else
        {
            spdlog::error(
                "[AssetManager] Could not kick binary load; missing record handle={}:{} path='{}'",
                handle.index,
                handle.generation,
                path.string());
        }
    }

    void AssetManager::commitCompletion(AssetCompletion&& completion)
    {
        AssetRecord* r = record(completion.handle);
        if (!r)
        {
            return;
        }

        {
            std::unique_lock lock(r->mutex);
            r->data = completion.success ? std::move(completion.data) : nullptr;
            r->error = std::move(completion.error);
        }

        r->state.store(
            completion.success ? AssetState::Loaded : AssetState::Failed,
            std::memory_order_release);
    }

    AssetManager::AssetRecord* AssetManager::record(AssetHandle handle)
    {
        return const_cast<AssetRecord*>(
            static_cast<const AssetManager&>(*this).record(handle));
    }

    const AssetManager::AssetRecord* AssetManager::record(AssetHandle handle) const
    {
        if (!handle)
        {
            return nullptr;
        }

        std::shared_lock lock(m_recordsMutex);
        if (handle.index >= m_records.size())
        {
            return nullptr;
        }

        const std::unique_ptr<AssetRecord>& r = m_records[handle.index];
        if (!r || r->handle.generation != handle.generation)
        {
            return nullptr;
        }

        return r.get();
    }

    std::filesystem::path AssetManager::resolvePath(const std::filesystem::path& path) const
    {
        if (path.is_absolute())
        {
            return path.lexically_normal();
        }

        return (m_desc.assetRoot / path).lexically_normal();
    }

    std::filesystem::path AssetManager::resolveExistingPath(
        const std::filesystem::path& path) const
    {
        const std::filesystem::path resolved = resolvePath(path);
        if (std::filesystem::exists(resolved))
        {
            return std::filesystem::weakly_canonical(resolved);
        }

        if (path.is_absolute())
        {
            return resolved;
        }

        std::filesystem::path cursor = std::filesystem::current_path();
        while (!cursor.empty())
        {
            const std::filesystem::path candidate =
                (cursor / path).lexically_normal();

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

        return resolved;
    }

    std::string AssetManager::canonicalKeyPath(const std::filesystem::path& path) const
    {
        std::error_code ec;
        std::filesystem::path canonical = std::filesystem::weakly_canonical(path, ec);
        if (ec)
        {
            canonical = path.lexically_normal();
        }

        return lowerCopy(canonical.generic_string());
    }

    AssetKind AssetManager::detectKind(const std::filesystem::path& path)
    {
        const std::string ext = lowerCopy(path.extension().string());

        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
            ext == ".bmp" || ext == ".tga" || ext == ".hdr")
        {
            return AssetKind::Image;
        }

        if (ext == ".gltf" || ext == ".glb")
        {
            return AssetKind::Model;
        }

        if (!ext.empty())
        {
            return AssetKind::Binary;
        }

        return AssetKind::Unknown;
    }

    uint64_t AssetManager::hashOptions(const ImageLoadOptions& options)
    {
        uint64_t h = 0;
        hashCombine(h, options.forceRGBA ? 1ull : 0ull);
        hashCombine(h, options.srgb ? 1ull : 0ull);
        hashCombine(h, options.flipY ? 1ull : 0ull);
        return h;
    }

    uint64_t AssetManager::hashOptions(const ModelLoadOptions& options)
    {
        uint64_t h = 0;
        hashCombine(h, options.loadMaterials ? 1ull : 0ull);
        hashCombine(h, options.loadTextures ? 1ull : 0ull);
        hashCombine(h, options.generateTangentsIfMissing ? 1ull : 0ull);
        hashCombine(h, options.mergeMeshes ? 1ull : 0ull);
        return h;
    }

    uint64_t AssetManager::hashOptions(const BinaryLoadOptions& options)
    {
        uint64_t h = 0;
        hashCombine(h, options.nullTerminate ? 1ull : 0ull);
        return h;
    }

    std::vector<std::byte> AssetManager::readWholeFile(
        const std::filesystem::path& path,
        bool nullTerminate)
    {
        return readWholeFileBytes(path, nullTerminate);
    }

    ImageAsset AssetManager::decodeImage(
        std::span<const std::byte> bytes,
        const ImageLoadOptions& options)
    {
        return decodeImageBytes(bytes, options);
    }

    ModelAsset AssetManager::decodeModel(
        const std::filesystem::path& path,
        const ModelLoadOptions& options)
    {
        auto data = fastgltf::GltfDataBuffer::FromPath(path);
        if (data.error() != fastgltf::Error::None)
        {
            throw std::runtime_error("fastgltf could not read file: " + path.string());
        }

        constexpr auto gltfOptions =
            fastgltf::Options::LoadExternalBuffers |
            fastgltf::Options::LoadExternalImages |
            fastgltf::Options::DecomposeNodeMatrices;

        constexpr fastgltf::Extensions supportedExtensions =
            fastgltf::Extensions::KHR_texture_transform |
            fastgltf::Extensions::KHR_mesh_quantization |
            fastgltf::Extensions::KHR_lights_punctual |
            fastgltf::Extensions::KHR_materials_sheen |
            fastgltf::Extensions::KHR_materials_specular |
            fastgltf::Extensions::KHR_materials_ior |
            fastgltf::Extensions::KHR_materials_clearcoat |
            fastgltf::Extensions::KHR_materials_emissive_strength |
            fastgltf::Extensions::KHR_materials_unlit |
            fastgltf::Extensions::KHR_materials_variants;


        fastgltf::Parser parser(supportedExtensions);
        auto assetResult = parser.loadGltf(
            data.get(),
            path.parent_path(),
            gltfOptions,
            fastgltf::Category::All);

        if (assetResult.error() != fastgltf::Error::None)
        {
            throw std::runtime_error(
                fmt::format(
                    "fastgltf parse failed for: '{}' error={}",
                    path.string(),
                    static_cast<int>(assetResult.error())));
        }

        fastgltf::Asset gltf = std::move(assetResult.get());

        ModelAsset model{};
        model.name = path.stem().string();

        model.samplers.reserve(gltf.samplers.size());
        for (const fastgltf::Sampler& src : gltf.samplers)
        {
            model.samplers.push_back(decodeSampler(src));
        }

        if (options.loadTextures)
        {
            model.images.reserve(gltf.images.size());
            for (size_t imageIndex = 0; imageIndex < gltf.images.size(); ++imageIndex)
            {
                const fastgltf::Image& src = gltf.images[imageIndex];
                std::vector<std::byte> imageBytes =
                    extractGltfImageBytes(gltf, src, path.parent_path());
                model.images.push_back(
                    decodeGltfImage(
                        imageBytes,
                        false,
                        src.name.empty()
                            ? std::string_view("image")
                            : std::string_view(src.name)));
            }
        }

        model.textures.reserve(gltf.textures.size());
        for (size_t textureIndex = 0; textureIndex < gltf.textures.size(); ++textureIndex)
        {
            const fastgltf::Texture& src = gltf.textures[textureIndex];

            TextureAsset dst{};
            dst.name = std::string(src.name);

            if (src.imageIndex.has_value())
            {
                dst.imageIndex = static_cast<int32_t>(*src.imageIndex);
            }
            else if (src.basisuImageIndex.has_value() ||
                     src.ddsImageIndex.has_value() ||
                     src.webpImageIndex.has_value())
            {
                spdlog::warn(
                    "[AssetManager] glTF texture {} uses a compressed/webp image path that is not decoded yet.",
                    textureIndex);
            }
            else
            {
                spdlog::warn(
                    "[AssetManager] glTF texture {} does not reference an image.",
                    textureIndex);
            }

            if (src.samplerIndex.has_value())
            {
                dst.samplerIndex = static_cast<int32_t>(*src.samplerIndex);
            }

            model.textures.push_back(std::move(dst));
        }

        if (options.loadMaterials)
        {
            model.materials.reserve(gltf.materials.size());
            for (const fastgltf::Material& src : gltf.materials)
            {
                MaterialAsset dst{};
                dst.name = std::string(src.name);
                dst.baseColorFactor = glm::vec4(
                    src.pbrData.baseColorFactor[0],
                    src.pbrData.baseColorFactor[1],
                    src.pbrData.baseColorFactor[2],
                    src.pbrData.baseColorFactor[3]);
                dst.metallicFactor = src.pbrData.metallicFactor;
                dst.roughnessFactor = src.pbrData.roughnessFactor;
                dst.alphaCutoff = src.alphaCutoff;
                dst.emissiveFactor = glm::vec3(
                    src.emissiveFactor[0],
                    src.emissiveFactor[1],
                    src.emissiveFactor[2]);
                dst.emissiveStrength = src.emissiveStrength;
                dst.doubleSided = src.doubleSided;
                dst.alphaBlend = src.alphaMode == fastgltf::AlphaMode::Blend;
                dst.alphaMask = src.alphaMode == fastgltf::AlphaMode::Mask;
                dst.unlit = src.unlit;

                if (src.pbrData.baseColorTexture.has_value())
                {
                    dst.baseColorTexture =
                        decodeTextureInfo(
                            *src.pbrData.baseColorTexture,
                            TextureTransferFunction::SRGB);
                }
                if (src.normalTexture.has_value())
                {
                    dst.normalTexture =
                        decodeNormalTextureInfo(*src.normalTexture);
                }
                if (src.pbrData.metallicRoughnessTexture.has_value())
                {
                    dst.metallicRoughnessTexture =
                        decodeTextureInfo(
                            *src.pbrData.metallicRoughnessTexture,
                            TextureTransferFunction::Linear);
                }
                if (src.occlusionTexture.has_value())
                {
                    dst.occlusionTexture =
                        decodeOcclusionTextureInfo(*src.occlusionTexture);
                }
                if (src.emissiveTexture.has_value())
                {
                    dst.emissiveTexture =
                        decodeTextureInfo(
                            *src.emissiveTexture,
                            TextureTransferFunction::SRGB);
                }

                model.materials.push_back(std::move(dst));
            }
        }

        bool modelBoundsInitialized = false;
        model.meshes.reserve(gltf.meshes.size());

        for (const fastgltf::Mesh& srcMesh : gltf.meshes)
        {
            MeshAsset dstMesh{};
            dstMesh.name = std::string(srcMesh.name);
            dstMesh.primitives.reserve(srcMesh.primitives.size());

            for (const fastgltf::Primitive& primitive : srcMesh.primitives)
            {
                if (primitive.type != fastgltf::PrimitiveType::Triangles)
                {
                    spdlog::warn(
                        "[AssetManager] Skipping non-triangle glTF primitive in '{}'.",
                        path.string());
                    continue;
                }

                const auto positionIt = primitive.findAttribute("POSITION");
                if (positionIt == primitive.attributes.end())
                {
                    spdlog::warn(
                        "[AssetManager] Skipping glTF primitive without POSITION in '{}'.",
                        path.string());
                    continue;
                }

                MeshPrimitiveAsset dstPrimitive{};
                dstPrimitive.firstVertex = static_cast<uint32_t>(model.vertices.size());
                dstPrimitive.firstIndex = static_cast<uint32_t>(model.indices.size());

                if (primitive.materialIndex.has_value())
                {
                    dstPrimitive.materialIndex =
                        static_cast<uint32_t>(*primitive.materialIndex);
                    if (options.loadMaterials &&
                        *primitive.materialIndex >= model.materials.size())
                    {
                        spdlog::warn(
                            "[AssetManager] glTF primitive references missing material {} in '{}'.",
                            *primitive.materialIndex,
                            path.string());
                    }
                }

                const fastgltf::Accessor& positionAccessor =
                    gltf.accessors[positionIt->accessorIndex];
                const size_t vertexCount = positionAccessor.count;

                std::vector<glm::vec3> positions(vertexCount);
                std::vector<glm::vec3> normals(vertexCount, glm::vec3(0.0f));
                std::vector<glm::vec4> tangents(vertexCount, glm::vec4(0.0f));
                std::vector<glm::vec2> uv0(vertexCount, glm::vec2(0.0f));
                std::vector<glm::vec2> uv1(vertexCount, glm::vec2(0.0f));
                std::vector<glm::vec4> colors(vertexCount, glm::vec4(1.0f));

                fastgltf::iterateAccessorWithIndex<glm::vec3>(
                    gltf,
                    positionAccessor,
                    [&](glm::vec3 value, size_t index) { positions[index] = value; });

                if (const auto normalIt = primitive.findAttribute("NORMAL");
                    normalIt != primitive.attributes.end())
                {
                    const fastgltf::Accessor& accessor =
                        gltf.accessors[normalIt->accessorIndex];
                    fastgltf::iterateAccessorWithIndex<glm::vec3>(
                        gltf,
                        accessor,
                        [&](glm::vec3 value, size_t index) { normals[index] = value; });
                }

                if (const auto tangentIt = primitive.findAttribute("TANGENT");
                    tangentIt != primitive.attributes.end())
                {
                    const fastgltf::Accessor& accessor =
                        gltf.accessors[tangentIt->accessorIndex];
                    fastgltf::iterateAccessorWithIndex<glm::vec4>(
                        gltf,
                        accessor,
                        [&](glm::vec4 value, size_t index) { tangents[index] = value; });
                }

                if (const auto uvIt = primitive.findAttribute("TEXCOORD_0");
                    uvIt != primitive.attributes.end())
                {
                    const fastgltf::Accessor& accessor =
                        gltf.accessors[uvIt->accessorIndex];
                    fastgltf::iterateAccessorWithIndex<glm::vec2>(
                        gltf,
                        accessor,
                        [&](glm::vec2 value, size_t index) { uv0[index] = value; });
                }

                if (const auto uvIt = primitive.findAttribute("TEXCOORD_1");
                    uvIt != primitive.attributes.end())
                {
                    const fastgltf::Accessor& accessor =
                        gltf.accessors[uvIt->accessorIndex];
                    fastgltf::iterateAccessorWithIndex<glm::vec2>(
                        gltf,
                        accessor,
                        [&](glm::vec2 value, size_t index) { uv1[index] = value; });
                }

                if (const auto colorIt = primitive.findAttribute("COLOR_0");
                    colorIt != primitive.attributes.end())
                {
                    const fastgltf::Accessor& accessor =
                        gltf.accessors[colorIt->accessorIndex];
                    fastgltf::iterateAccessorWithIndex<glm::vec4>(
                        gltf,
                        accessor,
                        [&](glm::vec4 value, size_t index) { colors[index] = value; });
                }

                model.vertices.reserve(model.vertices.size() + vertexCount);
                for (size_t i = 0; i < vertexCount; ++i)
                {
                    AssetVertex vertex{};
                    vertex.position = positions[i];
                    vertex.normal = normals[i];
                    vertex.tangent = tangents[i];
                    vertex.uv0 = uv0[i];
                    vertex.uv1 = uv1[i];
                    vertex.color = colors[i];
                    model.vertices.push_back(vertex);
                }

                dstPrimitive.vertexCount = static_cast<uint32_t>(vertexCount);
                dstPrimitive.bounds = boundsFromPositions(positions);
                expandBounds(model.bounds, dstPrimitive.bounds, modelBoundsInitialized);

                if (primitive.indicesAccessor.has_value())
                {
                    const fastgltf::Accessor& indexAccessor =
                        gltf.accessors[*primitive.indicesAccessor];

                    fastgltf::iterateAccessor<uint32_t>(
                        gltf,
                        indexAccessor,
                        [&](uint32_t index)
                        {
                            model.indices.push_back(dstPrimitive.firstVertex + index);
                        });
                }
                else
                {
                    for (uint32_t i = 0; i < dstPrimitive.vertexCount; ++i)
                    {
                        model.indices.push_back(dstPrimitive.firstVertex + i);
                    }
                }

                dstPrimitive.indexCount =
                    static_cast<uint32_t>(model.indices.size() - dstPrimitive.firstIndex);

                dstMesh.primitives.push_back(dstPrimitive);
            }

            model.meshes.push_back(std::move(dstMesh));
        }

        model.nodes.reserve(gltf.nodes.size());
        for (size_t nodeIndex = 0; nodeIndex < gltf.nodes.size(); ++nodeIndex)
        {
            const fastgltf::Node& srcNode = gltf.nodes[nodeIndex];

            NodeAsset dstNode{};
            dstNode.name = std::string(srcNode.name);

            if (srcNode.meshIndex.has_value())
            {
                dstNode.meshIndex = static_cast<int32_t>(*srcNode.meshIndex);
            }
            if (srcNode.cameraIndex.has_value())
            {
                dstNode.cameraIndex = static_cast<int32_t>(*srcNode.cameraIndex);
            }
            if (srcNode.lightIndex.has_value())
            {
                dstNode.lightIndex = static_cast<int32_t>(*srcNode.lightIndex);
            }
            if (srcNode.skinIndex.has_value())
            {
                dstNode.skinIndex = static_cast<int32_t>(*srcNode.skinIndex);
            }

            dstNode.localMatrix = toGlm(fastgltf::getTransformMatrix(srcNode));
            if (const auto* trs = std::get_if<fastgltf::TRS>(&srcNode.transform))
            {
                dstNode.translation = toGlm(trs->translation);
                dstNode.rotation = toGlm(trs->rotation);
                dstNode.scale = toGlm(trs->scale);
            }

            dstNode.children.reserve(srcNode.children.size());
            for (size_t childIndex : srcNode.children)
            {
                dstNode.children.push_back(static_cast<uint32_t>(childIndex));
            }

            model.nodes.push_back(std::move(dstNode));
        }

        for (size_t parentIndex = 0; parentIndex < gltf.nodes.size(); ++parentIndex)
        {
            for (size_t childIndex : gltf.nodes[parentIndex].children)
            {
                if (childIndex < model.nodes.size())
                {
                    model.nodes[childIndex].parent = static_cast<int32_t>(parentIndex);
                }
            }
        }

        model.scenes.reserve(gltf.scenes.size());
        for (const fastgltf::Scene& srcScene : gltf.scenes)
        {
            SceneAsset dstScene{};
            dstScene.name = std::string(srcScene.name);
            dstScene.rootNodes.reserve(srcScene.nodeIndices.size());
            for (size_t nodeIndex : srcScene.nodeIndices)
            {
                dstScene.rootNodes.push_back(static_cast<uint32_t>(nodeIndex));
            }
            model.scenes.push_back(std::move(dstScene));
        }

        model.defaultScene =
            gltf.defaultScene.has_value()
                ? static_cast<int32_t>(*gltf.defaultScene)
                : (model.scenes.empty() ? -1 : 0);

        if (!gltf.scenes.empty())
        {
            const size_t sceneIndex =
                model.defaultScene >= 0
                    ? static_cast<size_t>(model.defaultScene)
                    : 0;
            fastgltf::iterateSceneNodes(
                gltf,
                sceneIndex,
                fastgltf::math::fmat4x4(),
                [&](fastgltf::Node& node, const fastgltf::math::fmat4x4& world)
                {
                    const ptrdiff_t index = &node - gltf.nodes.data();
                    if (index >= 0 && static_cast<size_t>(index) < model.nodes.size())
                    {
                        model.nodes[static_cast<size_t>(index)].worldMatrix = toGlm(world);
                    }
                });
        }

        for (size_t materialIndex = 0; materialIndex < model.materials.size(); ++materialIndex)
        {
            const MaterialAsset& material = model.materials[materialIndex];
            const MaterialTextureSlot slots[] = {
                material.baseColorTexture,
                material.normalTexture,
                material.metallicRoughnessTexture,
                material.occlusionTexture,
                material.emissiveTexture,
            };

            for (const MaterialTextureSlot& slot : slots)
            {
                if (slot.textureIndex >= 0 &&
                    static_cast<size_t>(slot.textureIndex) >= model.textures.size())
                {
                    spdlog::warn(
                        "[AssetManager] glTF material {} references missing texture {} in '{}'.",
                        materialIndex,
                        slot.textureIndex,
                        path.string());
                }
            }
        }

        spdlog::info(
            "[AssetManager] Loaded glTF '{}': {} vertices, {} indices, {} meshes, {} materials, {} images, {} textures, {} samplers, {} nodes, {} scenes",
            path.string(),
            model.vertices.size(),
            model.indices.size(),
            model.meshes.size(),
            model.materials.size(),
            model.images.size(),
            model.textures.size(),
            model.samplers.size(),
            model.nodes.size(),
            model.scenes.size());

        return model;
    }
}
