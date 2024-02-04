#include <chrono>
#include <stdexcept>
#include <utility>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION

#include <stb_image.h>
#include <glfw3.h>

#include <graphicsManager.h>

#if defined(TESTPOS)
    #include "testPos.h"
#elif defined(TESTCUDA)
    #include "testCuda.h"
#else
    #include "testScene.h"
#endif

bool framebufferResized = false;
uint32_t imageCount = 3;

VkPhysicalDeviceFeatures physicalDeviceFeatures(){
    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    deviceFeatures.independentBlend = VK_TRUE;
    deviceFeatures.sampleRateShading = VK_TRUE;
    deviceFeatures.imageCubeArray = VK_TRUE;
    deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;
    deviceFeatures.fillModeNonSolid = VK_TRUE;
    return deviceFeatures;
}

GLFWwindow* initializeWindow(uint32_t WIDTH, uint32_t HEIGHT, std::filesystem::path iconName = "");
std::pair<uint32_t,uint32_t> resize(GLFWwindow* window, graphicsManager* app, scene* testScene);

using clk = std::chrono::high_resolution_clock;
template<typename type> type period(clk::time_point time){
    return std::chrono::duration<type, std::chrono::seconds::period>(clk::now() - time).count();
}

int main()
{
    uint32_t WIDTH = 800;
    uint32_t HEIGHT = 800;
    const std::filesystem::path ExternalPath = std::filesystem::absolute(std::filesystem::path(__FILE__).replace_filename("../../"));

    GLFWwindow* window = initializeWindow(WIDTH, HEIGHT, ExternalPath / "dependences/texture/icon.PNG");

    graphicsManager app(window, imageCount, physicalDeviceFeatures(), 1);

#if defined(TESTPOS)
    testPos testScene(&app, window, ExternalPath);
#elif defined(TESTCUDA)
    testCuda testScene(&app, window, ExternalPath);
#else
    testScene testScene(&app, window, ExternalPath, framebufferResized);
#endif
    testScene.create(WIDTH,HEIGHT);

    //Memory::instance().status();

    for(float time = 1.0f; !glfwWindowShouldClose(window);){
        if(auto start = clk::now(); app.checkNextFrame() != VK_ERROR_OUT_OF_DATE_KHR) {
            testScene.updateFrame(app.getImageIndex(), time);

            if (VkResult result = app.drawFrame(); result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized){
                std::tie(WIDTH, HEIGHT) = resize(window,&app,&testScene);
            } else if(result) {
                throw std::runtime_error("failed to with " + std::to_string(result));
            }
            time = period<float>(start);
        }
    }

    debug::checkResult(app.deviceWaitIdle(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));

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
    app->destroy();

    app->create(window, imageCount);

    testScene->resize(width, height);

    //Memory::instance().status();

    framebufferResized = false;

    return std::pair<uint32_t,uint32_t>(width, height);
}

GLFWwindow* initializeWindow(uint32_t WIDTH, uint32_t HEIGHT, std::filesystem::path iconName)
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan render", nullptr, nullptr);
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
