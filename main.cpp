#include "core/vulkanCore.h"
#include "core/texture.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/light.h"
#include "core/transformational/object.h"
#include "core/transformational/group.h"
#include "core/transformational/camera.h"
#include "libs/stb-master/stb_image.h"

#include <chrono>
#include <stdexcept>        // предотвращения ошибок
#include <cstdlib>          // заголовок для использования макросов EXIT_SUCCESS и EXIT_FAILURE
#include <sstream>

uint32_t WIDTH = 800;
uint32_t HEIGHT = 800;

bool framebufferResized = false;

bool updateCamera = false;
float cameraAngle = 45.0f;

GLFWwindow* window;

camera *cameras;
texture *emptyTexture;
texture *emptyTextureW;

std::string ZERO_TEXTURE        = ExternalPath + "texture\\0.png";
std::string ZERO_TEXTURE_WHITE  = ExternalPath + "texture\\1.png";

std::vector<std::string> SKYBOX = {
    ExternalPath+"texture\\skybox\\left.jpg",
    ExternalPath+"texture\\skybox\\right.jpg",
    ExternalPath+"texture\\skybox\\front.jpg",
    ExternalPath+"texture\\skybox\\back.jpg",
    ExternalPath+"texture\\skybox\\top.jpg",
    ExternalPath+"texture\\skybox\\bottom.jpg"
};

void framebufferResizeCallback(GLFWwindow* window, int width, int height);
void initializeWindow(GLFWwindow* &window);
void recreateSwapChain(VkApplication* app, GLFWwindow* window);

#define TestScene1
#include "testScene1.cpp"
#include "testScene2.cpp"

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

    cameras = new camera;
        cameras->translate(glm::vec3(0.0f,0.0f,10.0f));
        glm::mat4x4 proj = glm::perspective(glm::radians(cameraAngle), (float) WIDTH / (float) HEIGHT, 0.1f, 1000.0f);
        proj[1][1] *= -1.0f;
        cameras->setProjMatrix(proj);
    app.setCameraObject(cameras);

    emptyTexture = new texture(ZERO_TEXTURE);
    emptyTextureW = new texture(ZERO_TEXTURE_WHITE);
    app.setEmptyTexture(emptyTexture);

    app.createGraphics(window);

    createScene(&app);

    app.updateDescriptorSets();
    app.createCommandBuffers();
    app.createSyncObjects();

    app.resetUboWorld();
    app.resetUboLight();

    runScene(&app,window);

    app.deviceWaitIdle();

    destroyScene(&app);
    app.cleanup();

    delete emptyTexture;
    delete emptyTextureW;

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
