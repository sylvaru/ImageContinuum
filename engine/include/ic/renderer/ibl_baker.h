#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "ic/core/asset_manager.h"
#include "ic/renderer/render_handles.h"
#include "ic/renderer/render_types.h"

namespace ic
{
    struct IBLHandle
    {
        uint32_t index = UINT32_MAX;
        uint32_t generation = 0;

        explicit operator bool() const
        {
            return index != UINT32_MAX;
        }

        friend bool operator==(IBLHandle lhs, IBLHandle rhs)
        {
            return lhs.index == rhs.index &&
                   lhs.generation == rhs.generation;
        }

        friend bool operator!=(IBLHandle lhs, IBLHandle rhs)
        {
            return !(lhs == rhs);
        }
    };

    enum class IBLBakeState : uint8_t
    {
        Empty = 0,
        Queued,
        WaitingForSource,
        UploadingGPU,
        Baking,
        Ready,
        Failed
    };

    struct IBLBakeDesc
    {
        AssetHandle sourceEnvironment = {};
        uint32_t environmentSize = 1024;
        uint32_t irradianceSize = 64;
        uint32_t prefilterSize = 256;
        uint32_t brdfLutSize = 512;
        TextureFormat format = TextureFormat::RGBA32_Float;
        std::string debugName;
    };

    struct IBLResources
    {
        TextureHandle environmentCubemap = {};
        TextureHandle irradianceCubemap = {};
        TextureHandle prefilteredCubemap = {};
        TextureHandle brdfLut = {};

        uint32_t environmentBindlessIndex = UINT32_MAX;
        uint32_t irradianceBindlessIndex = UINT32_MAX;
        uint32_t prefilteredBindlessIndex = UINT32_MAX;
        uint32_t brdfLutBindlessIndex = UINT32_MAX;

        uint32_t prefilteredMipCount = 1;
        float intensity = 1.0f;
    };

    struct SceneEnvironment
    {
        IBLHandle ibl = {};
        float exposure = 1.0f;
        float skyIntensity = 1.0f;
        bool visible = true;
    };

    struct GPUSceneEnvironment
    {
        uint32_t environmentTextureIndex = UINT32_MAX;
        uint32_t irradianceTextureIndex = UINT32_MAX;
        uint32_t prefilteredTextureIndex = UINT32_MAX;
        uint32_t brdfLutTextureIndex = UINT32_MAX;

        uint32_t prefilteredMipCount = 1;
        float environmentIntensity = 1.0f;
        float exposure = 1.0f;
        float padding = 0.0f;
    };

    struct IBLBakeRequest
    {
        uint64_t requestId = 0;
        IBLHandle handle = {};
        IBLBakeDesc desc = {};
    };

    struct IBLBakeResult
    {
        uint64_t requestId = 0;
        IBLHandle handle = {};
        IBLResources resources = {};
        bool success = false;
        std::string error;
    };

    struct IBLProbeSnapshot
    {
        IBLHandle handle = {};
        IBLBakeDesc desc = {};
        IBLResources resources = {};
        IBLBakeState state = IBLBakeState::Empty;
        std::string error;
        uint64_t requestId = 0;
    };

    class IBLBaker final
    {
    public:
        IBLHandle requestBake(const IBLBakeDesc& desc);
        IBLHandle requestRebake(IBLHandle previous, const IBLBakeDesc& desc);

        [[nodiscard]] IBLBakeState state(IBLHandle handle) const;
        [[nodiscard]] bool resources(
            IBLHandle handle,
            IBLResources& outResources) const;
        [[nodiscard]] IBLProbeSnapshot snapshot(IBLHandle handle) const;

        std::vector<IBLBakeRequest> takePendingRequests();
        bool requeue(IBLHandle handle);
        void markUploading(IBLHandle handle, uint64_t requestId);
        void markBaking(IBLHandle handle, uint64_t requestId);
        void markWaitingForSource(
            IBLHandle handle,
            uint64_t requestId);
        void publishResult(const IBLBakeResult& result);

    private:
        struct ProbeRecord
        {
            uint32_t generation = 1;
            IBLBakeDesc desc = {};
            IBLResources resources = {};
            IBLBakeState state = IBLBakeState::Empty;
            std::string error;
            uint64_t requestId = 0;
        };

        [[nodiscard]] ProbeRecord* record(IBLHandle handle);
        [[nodiscard]] const ProbeRecord* record(IBLHandle handle) const;
        [[nodiscard]] static bool sameBakeKey(
            const IBLBakeDesc& lhs,
            const IBLBakeDesc& rhs);
        [[nodiscard]] IBLHandle enqueueLocked(
            uint32_t index,
            const IBLBakeDesc& desc);

        mutable std::mutex m_mutex;
        std::vector<ProbeRecord> m_records;
        std::vector<uint32_t> m_freeList;
        std::vector<IBLBakeRequest> m_pendingRequests;
        uint64_t m_nextRequestId = 1;
    };
}
