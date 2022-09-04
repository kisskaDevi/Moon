#include "core/vulkanCore.h"

#include <chrono>
#include <stdexcept>        // предотвращения ошибок
#include <cstdlib>          // заголовок для использования макросов EXIT_SUCCESS и EXIT_FAILURE
#include <sstream>

uint32_t WIDTH = 800;
uint32_t HEIGHT = 800;

bool framebufferResized = false;

float fps = 60.0f;
bool  fpsLock = false;

GLFWwindow* window;

const std::string ExternalPath = "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\";
std::string ZERO_TEXTURE        = ExternalPath + "texture\\0.png";
std::string ZERO_TEXTURE_WHITE  = ExternalPath + "texture\\1.png";


void framebufferResizeCallback(GLFWwindow* window, int width, int height);
void initializeWindow(GLFWwindow* &window);
void recreateSwapChain(VkApplication* app, GLFWwindow* window);

#include "scene.h"

int main()
{
    initializeWindow(window);

    VkApplication app;
    app.createInstance();
    app.setupDebugMessenger();
    app.createSurface(window);
    app.pickPhysicalDevice();
    app.createLogicalDevice();
    app.checkSwapChainSupport();
    app.createCommandPool();
    app.createGraphics(window);

    app.setEmptyTexture(ZERO_TEXTURE);

    scene testScene;

    testScene.createScene(&app,WIDTH,HEIGHT);

    app.updateDescriptorSets();
    app.createCommandBuffers();
    app.createSyncObjects();

    app.resetUboWorld();
    app.resetUboLight();

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
            testScene.updateFrame(&app,window,app.getImageIndex(),frameTime,WIDTH,HEIGHT);

            VkResult result = app.drawFrame();

            if (result == VK_ERROR_OUT_OF_DATE_KHR)                         recreateSwapChain(&app,window);
            else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)   throw std::runtime_error("failed to acquire swap chain image!");

            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized){
                framebufferResized = false;
                recreateSwapChain(&app,window);
            }else if(result != VK_SUCCESS){
                throw std::runtime_error("failed to present swap chain image!");
            }
        }
    }

    app.deviceWaitIdle();

    testScene.destroyScene(&app);
    app.cleanup();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

void recreateSwapChain(VkApplication *app, GLFWwindow* window)
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

    app->destroyGraphics();
    app->freeCommandBuffers();

    app->checkSwapChainSupport();
    app->createGraphics(window);
    app->updateDescriptorSets();
    app->createCommandBuffers();

    app->resetUboWorld();
    app->resetUboLight();
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
    glfwInit();                                                             //инициализация библиотеки GLFW
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);                           //указывает не создавать контекст OpenGL (GLFW изначально был разработан для создания контекста OpenGL)

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);   //инициализация собственного окна
    glfwSetWindowUserPointer(window, nullptr);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    int width,height,comp;
    std::string filename = ExternalPath + "texture\\icon.png";
    stbi_uc* img = stbi_load(filename.c_str(), &width, &height, &comp, 0);
    GLFWimage images{width,height,img};
    glfwSetWindowIcon(window,1,&images);
}
