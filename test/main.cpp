#ifdef WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#define STB_IMAGE_IMPLEMENTATION

#include "graphicsManager.h"
#include "testScene.h"
#include "testpos.h"

#include "stb_image.h"
#include <glfw3.h>
#include <chrono>
#include <stdexcept>
#include <cstdlib>
#include <sstream>
#include <utility>
#include <filesystem>

#include <vector.h>
#include <matrix.h>

bool framebufferResized = false;

GLFWwindow* initializeWindow(uint32_t WIDTH, uint32_t HEIGHT, std::filesystem::path iconName = "");
std::pair<uint32_t,uint32_t> resize(GLFWwindow* window, graphicsManager* app, scene* testScene);

int main()
{
    float fps = 60.0f;
    bool fpsLock = false;
    uint32_t WIDTH = 800;
    uint32_t HEIGHT = 800;
    const std::filesystem::path ExternalPath = std::filesystem::absolute(std::string(__FILE__) + "/../../");

    GLFWwindow* window = initializeWindow(WIDTH, HEIGHT, ExternalPath / "dependences/texture/icon.png");

    graphicsManager app;
    debug::checkResult(app.createSurface(window), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(app.createDevice(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(app.createSwapChain(window), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(app.createSyncObjects(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));

    testScene testScene(&app, window, ExternalPath);
    testScene.create(WIDTH,HEIGHT);

    static auto pastTime = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window))
    {
        float frameTime = std::chrono::duration<float, std::chrono::seconds::period>(std::chrono::high_resolution_clock::now() - pastTime).count();

        if(fpsLock && fps < 1.0f/frameTime) continue;
        pastTime = std::chrono::high_resolution_clock::now();

        glfwSetWindowTitle(window, std::stringstream("Vulkan [" + std::to_string(1.0f/frameTime) + " FPS]").str().c_str());

        if(app.checkNextFrame() != VK_ERROR_OUT_OF_DATE_KHR)
        {
            testScene.updateFrame(app.getImageIndex(),frameTime,WIDTH,HEIGHT);

            if (VkResult result = app.drawFrame(); result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized){
                framebufferResized = false;
                std::tie(WIDTH, HEIGHT) = resize(window,&app,&testScene);
            } else if(result) {
                throw std::runtime_error("failed to with " + std::to_string(result));
            }
        }
    }

    debug::checkResult(app.deviceWaitIdle(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));

    testScene.destroy();
    app.destroySwapChain();
    app.destroy();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

std::pair<uint32_t,uint32_t> resize(GLFWwindow* window, graphicsManager* app, scene* testScene)
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width * height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    app->deviceWaitIdle();
    app->destroySwapChain();
    app->createSwapChain(window);

    testScene->resize(width, height);

    return std::pair<uint32_t,uint32_t>(width, height);
}

GLFWwindow* initializeWindow(uint32_t WIDTH, uint32_t HEIGHT, std::filesystem::path iconName)
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, nullptr);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int, int){ framebufferResized = true;});

    if(iconName.string().size() > 0){
        int width, height, comp;
        stbi_uc* img = stbi_load(iconName.string().c_str(), &width, &height, &comp, 0);
        GLFWimage images{width,height,img};
        glfwSetWindowIcon(window,1,&images);
    }

    return window;
}
