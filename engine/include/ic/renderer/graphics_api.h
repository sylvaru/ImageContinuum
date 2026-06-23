// graphics_api.h
#pragma once
#include "ic/interface/window.h"
#include "ic/renderer/renderer_backend.h"

namespace ic
{

    struct FrameData {
        //alignas(256) Matrix4 model;
        //alignas(256) Matrix4 viewProj;
        //alignas(16)  Vector4 cameraPos;
        alignas(4)   uint32_t materialIndex;
    };


	//std::unique_ptr<RendererBackend>
	//	createRenderer(Window& window, const RendererSpecification& spec);
}


/*


*/