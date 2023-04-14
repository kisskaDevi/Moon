#include "../core/graphicsManager.h"
#include "../core/deferredGraphics/deferredGraphics.h"
#include "../core/transformational/baseCamera.h"
#include <glfw3.h>
#include <glm.hpp>

#include <chrono>
#include <stdexcept>
#include <cstdlib>
#include <sstream>

#include "scene.h"
#include "stb_image.h"

uint32_t WIDTH = 800;
uint32_t HEIGHT = 800;

bool framebufferResized = false;

float fps = 60.0f;
bool  fpsLock = false;

GLFWwindow* window;

const std::string ExternalPath = "C:\\Qt\\repositories\\kisskaVulkan\\";

const std::string ZERO_TEXTURE        = ExternalPath + "dependences\\texture\\0.png";
const std::string ZERO_TEXTURE_WHITE  = ExternalPath + "dependences\\texture\\1.png";

void framebufferResizeCallback(GLFWwindow* window, int width, int height);
void initializeWindow(GLFWwindow* &window);
void recreateSwapChain(graphicsManager* app, deferredGraphics* graphics, GLFWwindow* window, baseCamera* cameraObject);

int main()
{
    initializeWindow(window);

    deferredGraphics graphics(ExternalPath);

    graphicsManager app;
    app.createInstance();
    app.createSurface(window);
    app.createDevice();
    app.setGraphics(&graphics);
    graphics.createCommandPool();

    baseCamera cameraObject;
        cameraObject.translate(glm::vec3(0.0f,0.0f,10.0f));
        glm::mat4x4 proj = glm::perspective(glm::radians(90.0f), (float) WIDTH / (float) HEIGHT, 0.1f, 500.0f);
        proj[1][1] *= -1.0f;
        cameraObject.setProjMatrix(proj);
    graphics.bindCameraObject(&cameraObject);

    graphics.setEmptyTexture(ZERO_TEXTURE);

    app.createGraphics(window);
    graphics.updateDescriptorSets();

    scene testScene(&app,&graphics,window,ExternalPath);
    testScene.createScene(WIDTH,HEIGHT,&cameraObject);

    app.createCommandBuffers();
    app.createSyncObjects();

    static auto pastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window))
    {
        auto currentTime = std::chrono::high_resolution_clock::now();
        float frameTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - pastTime).count();

            if(fpsLock)
                if(fps<1.0f/frameTime)  continue;
            pastTime = currentTime;

            std::stringstream ss;
            ss << "Vulkan" << " [" << 1.0f/frameTime << " FPS]";
            glfwSetWindowTitle(window, ss.str().c_str());

        if(app.checkNextFrame()!=VK_ERROR_OUT_OF_DATE_KHR)
        {
            testScene.updateFrame(app.getImageIndex(),frameTime,WIDTH,HEIGHT);

            VkResult result = app.drawFrame();

            if (result == VK_ERROR_OUT_OF_DATE_KHR)                         recreateSwapChain(&app,&graphics,window,&cameraObject);
            else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)   throw std::runtime_error("failed to acquire swap chain image!");

            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized || testScene.framebufferResized){
                framebufferResized = false;
                testScene.framebufferResized = false;
                recreateSwapChain(&app,&graphics,window,&cameraObject);
            }else if(result != VK_SUCCESS){
                throw std::runtime_error("failed to present swap chain image!");
            }
        }
    }

    app.deviceWaitIdle();

    graphics.removeCameraObject(&cameraObject);
    testScene.destroyScene();
    graphics.destroyGraphics();
    app.destroy();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

void recreateSwapChain(graphicsManager* app, deferredGraphics* graphics, GLFWwindow* window, baseCamera* cameraObject)
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
    graphics->destroyGraphics();

    glm::mat4x4 proj = glm::perspective(glm::radians(90.0f), (float) WIDTH / (float) HEIGHT, 0.1f, 500.0f);
    proj[1][1] *= -1.0f;
    cameraObject->setProjMatrix(proj);

    graphics->setExtent({static_cast<uint32_t>(width),static_cast<uint32_t>(height)});
    app->createGraphics(window);
    graphics->updateDescriptorSets();
    app->createCommandBuffers();
}

void framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    static_cast<void>(width);
    static_cast<void>(height);
    static_cast<void>(window);
    framebufferResized = true;
}
void initializeWindow(GLFWwindow* &window)
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, nullptr);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    int width,height,comp;
    std::string filename = ExternalPath + "dependences\\texture\\icon.png";
    stbi_uc* img = stbi_load(filename.c_str(), &width, &height, &comp, 0);
    GLFWimage images{width,height,img};
    glfwSetWindowIcon(window,1,&images);
}
