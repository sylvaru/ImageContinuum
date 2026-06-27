//  ic/platform/glfw_context.h
#pragma once
#include <GLFW/glfw3.h> 

namespace ic
{
    struct GLFWContext
    {
        GLFWContext()
        {
            if (!glfwInit())
                throw std::runtime_error("Failed to initialize GLFW");
        }

        ~GLFWContext()
        {
            glfwTerminate();
        }

        GLFWContext(const GLFWContext&) = delete;
        GLFWContext& operator=(const GLFWContext&) = delete;
    };
}