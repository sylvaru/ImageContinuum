#pragma once

#include <d3d12.h>

namespace ic
{
    // Records an all-subresources transition barrier, no-op when the resource is
    // null or already in the target state. Shared by the backend's barrier
    // application and by the pass recorders that own a pass-internal transition
    // (e.g. the Hi-Z mip chain). State tracking itself stays with the caller.
    inline void transitionResource(
        ID3D12GraphicsCommandList4* cmd,
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES before,
        D3D12_RESOURCE_STATES after)
    {
        if (!resource || before == after)
        {
            return;
        }
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource =
            D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;
        cmd->ResourceBarrier(1, &barrier);
    }
}
