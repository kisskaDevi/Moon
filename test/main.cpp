#define VK_USE_PLATFORM_WIN32_KHR
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

uint32_t WIDTH = 800;
uint32_t HEIGHT = 800;

bool framebufferResized = false;

void framebufferResizeCallback(GLFWwindow* window, int width, int height);
void initializeWindow(GLFWwindow* &window, uint32_t WIDTH, uint32_t HEIGHT, std::string iconName);
void recreateSwapChain(graphicsManager* app, std::vector<deferredGraphics*> graphics, GLFWwindow* window, baseCamera* cameraObject);

int main()
{
    float fps = 60.0f;
    bool fpsLock = false;
    const std::string ExternalPath = "C:\\Qt\\repositories\\kisskaVulkan\\";

    GLFWwindow* window;
    initializeWindow(window, WIDTH, HEIGHT, ExternalPath + "dependences\\texture\\icon.png");

    std::vector<deferredGraphics*> graphics = {
        new deferredGraphics{ExternalPath, {0.0f, 0.0f}, {0.5f, 1.0f}}
        , new deferredGraphics{ExternalPath, {0.5f, 0.0f}, {0.5f, 1.0f}}
    };

    graphicsManager app;
    app.createInstance();
    app.createSurface(window);
    app.createDevice();
    app.createSwapChain(window);
    for(auto& graphics: graphics){
        app.setGraphics(graphics);
        graphics->createCommandPool();
    }

    baseCamera cameraObject(45.0f, 0.5f * (float) WIDTH / (float) HEIGHT, 0.1f, 500.0f);
    cameraObject.translate(glm::vec3(0.0f,0.0f,10.0f));

    for(auto& graph: graphics){
        graph->bindCameraObject(&cameraObject, &graph == &graphics[0]);
    }

    const std::string ZERO_TEXTURE        = ExternalPath + "dependences\\texture\\0.png";
    const std::string ZERO_TEXTURE_WHITE  = ExternalPath + "dependences\\texture\\1.png";
    for(auto& graphics: graphics){
        graphics->setEmptyTexture(ZERO_TEXTURE);
    }
    app.createGraphics(window);
    for(auto& graphics: graphics){
        graphics->updateDescriptorSets();
    }

    scene testScene(&app, graphics, window, ExternalPath);
    testScene.createScene(WIDTH,HEIGHT,&cameraObject);

    app.createCommandBuffers();
    app.createSyncObjects();

    static auto pastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window))
    {
        auto currentTime = std::chrono::high_resolution_clock::now();
        float frameTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - pastTime).count();

        if(fpsLock && fps < 1.0f/frameTime)  continue;
        pastTime = currentTime;

        std::stringstream ss;
        ss << "Vulkan" << " [" << 1.0f/frameTime << " FPS]";
        glfwSetWindowTitle(window, ss.str().c_str());

        if(app.checkNextFrame()!=VK_ERROR_OUT_OF_DATE_KHR)
        {
            testScene.updateFrame(app.getImageIndex(),frameTime,WIDTH,HEIGHT);

            if (VkResult result = app.drawFrame(); result == VK_ERROR_OUT_OF_DATE_KHR){
                recreateSwapChain(&app,graphics,window,&cameraObject);
            } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR){
                throw std::runtime_error("failed to acquire swap chain image!");
            } else {
                if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized || testScene.framebufferResized){
                    framebufferResized = false;
                    testScene.framebufferResized = false;
                    recreateSwapChain(&app,graphics,window,&cameraObject);
                }else if(result != VK_SUCCESS){
                    throw std::runtime_error("failed to present swap chain image!");
                }
            }
        }
    }

    app.deviceWaitIdle();

    for(auto& graphics: graphics){
        graphics->removeCameraObject(&cameraObject);
        graphics->destroyGraphics();
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

void recreateSwapChain(graphicsManager* app, std::vector<deferredGraphics*> graphics, GLFWwindow* window, baseCamera* cameraObject)
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }
    WIDTH = width;
    HEIGHT = height;
    app->deviceWaitIdle();
    app->destroySwapChain();
    for(auto graphics: graphics){
        graphics->destroyGraphics();
    }

    cameraObject->recreate(45.0f, 0.5f * (float) WIDTH / (float) HEIGHT, 0.1f, 500.0f);

    app->createSwapChain(window);
    app->createGraphics(window);
    for(auto graphics: graphics){
        graphics->updateDescriptorSets();
    }
    app->createCommandBuffers();
}

void framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    static_cast<void>(width);
    static_cast<void>(height);
    static_cast<void>(window);
    framebufferResized = true;
}
void initializeWindow(GLFWwindow* &window, uint32_t WIDTH, uint32_t HEIGHT, std::string iconName)
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, nullptr);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    int width,height,comp;
    stbi_uc* img = stbi_load(iconName.c_str(), &width, &height, &comp, 0);
    GLFWimage images{width,height,img};
    glfwSetWindowIcon(window,1,&images);
}
