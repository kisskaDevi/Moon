#ifdef WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#define STB_IMAGE_IMPLEMENTATION

#include "graphicsManager.h"
#include "deferredGraphics.h"
#include "baseCamera.h"
#include "scene.h"

#include "stb_image.h"
#include <glfw3.h>
#include <glm.hpp>
#include <chrono>
#include <stdexcept>
#include <cstdlib>
#include <sstream>
#include <utility>
#include <filesystem>

bool framebufferResized = false;

void initializeWindow(GLFWwindow* &window, uint32_t WIDTH, uint32_t HEIGHT, std::filesystem::path iconName);
std::pair<uint32_t,uint32_t> recreateSwapChain(GLFWwindow* window, graphicsManager* app, std::vector<deferredGraphics*> graphics, baseCamera* cameraObject);

int main()
{
    GLFWwindow* window = nullptr;
    float fps = 60.0f;
    bool fpsLock = false;
    uint32_t WIDTH = 800;
    uint32_t HEIGHT = 800;
    const std::filesystem::path ExternalPath = std::filesystem::absolute(std::string(__FILE__) + "/../../");

    initializeWindow(window, WIDTH, HEIGHT, ExternalPath / "dependences/texture/icon.png");

    std::vector<deferredGraphics*> graphics = {
          new deferredGraphics{ExternalPath / "core/deferredGraphics/shaders", {WIDTH/2, HEIGHT}}
        , new deferredGraphics{ExternalPath / "core/deferredGraphics/shaders", {WIDTH/2, HEIGHT}, {static_cast<int32_t>(WIDTH / 2), 0}}
    };

    graphicsManager app;
    app.createDevice();
    app.createSurface(window);
    app.createSwapChain(window);
    app.createSyncObjects();

    baseCamera cameraObject(45.0f, 0.5f * (float) WIDTH / (float) HEIGHT, 0.1f, 500.0f);
    cameraObject.translate(glm::vec3(0.0f,0.0f,10.0f));

    for(auto& graph: graphics){
        app.setGraphics(graph);
        graph->setEmptyTexture(ExternalPath / "dependences/texture/0.png");
        graph->bindCameraObject(&cameraObject, &graph == &graphics[0]);
        graph->createGraphics(window, &app.getSurface());
        graph->updateDescriptorSets();
        graph->createCommandBuffers();
        graph->updateCommandBuffers();
    }

    scene testScene(&app, graphics, window, ExternalPath);
    testScene.createScene(WIDTH,HEIGHT,&cameraObject);

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
                std::tie(WIDTH, HEIGHT) = recreateSwapChain(window,&app,graphics,&cameraObject);
            } else if(result) {
                throw std::runtime_error("failed to with " + std::to_string(result));
            }
        }
    }

    app.deviceWaitIdle();

    for(auto& graph: graphics){
        graph->removeCameraObject(&cameraObject);
        graph->destroyGraphics();
    }
    testScene.destroyScene();
    app.destroySwapChain();
    app.destroy();

    for(auto& graphics: graphics){
        delete graphics;
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

std::pair<uint32_t,uint32_t> recreateSwapChain(GLFWwindow* window, graphicsManager* app, std::vector<deferredGraphics*> graphics, baseCamera* cameraObject)
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width * height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    cameraObject->recreate(45.0f, 0.5f * (float) width / (float) height, 0.1f, 500.0f);
    graphics[0]->setExtentAndOffset({static_cast<uint32_t>(width / 2), static_cast<uint32_t>(height)});
    graphics[1]->setExtentAndOffset({static_cast<uint32_t>(width / 2), static_cast<uint32_t>(height)}, {static_cast<int32_t>(width / 2), 0});

    app->deviceWaitIdle();
    app->destroySwapChain();
    app->createSwapChain(window);

    for(auto& graph: graphics){
        graph->destroyGraphics();
        graph->createGraphics(window, &app->getSurface());
        graph->updateDescriptorSets();
        graph->createCommandBuffers();
        graph->updateCommandBuffers();
    }

    return std::pair<uint32_t,uint32_t>(width, height);
}

void initializeWindow(GLFWwindow* &window, uint32_t WIDTH, uint32_t HEIGHT, std::filesystem::path iconName)
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, nullptr);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int, int){ framebufferResized = true;});

    int width, height, comp;
    stbi_uc* img = stbi_load(iconName.string().c_str(), &width, &height, &comp, 0);
    GLFWimage images{width,height,img};
    glfwSetWindowIcon(window,1,&images);
}
