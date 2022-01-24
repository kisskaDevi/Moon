#ifndef VULKANCORE_H
#define VULKANCORE_H

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3native.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <libs/glm/glm/glm.hpp>
#include <libs/glm/glm/gtc/matrix_transform.hpp>

class texture;
class cubeTexture;
class object;
class group;
class camera;
class spotLight;
class pointLight;
struct gltfModel;

template <typename type>
class light;

template <> class light<spotLight>;
template <> class light<pointLight>;

#include <chrono>

#include <iostream>         // заголовки для
#include <stdexcept>        // предотвращения ошибок
#include <cstdlib>          // заголовок для использования макросов EXIT_SUCCESSи EXIT_FAILURE
#include <vector>
#include <set>
#include <cstdint>          // нужна для UINT32_MAX
#include <algorithm>        // нужна для std::min/std::max
#include <fstream>
#include <array>
#include "graphics/attachments.h"
#include "graphics/graphics.h"

const std::string ExternalPath = "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\";

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 800;
const double pi = 4*std::atan(1);

const int MAX_FRAMES_IN_FLIGHT = 3;
const int COMMAND_POOLS = 1;

const std::vector<const char*> validationLayers = {
  "VK_LAYER_KHRONOS_validation"
};

//список необходимых расширений устройств, аналогичный списку уровней проверки
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

typedef struct physicalDevice{
    VkPhysicalDevice device;
    std::vector<QueueFamilyIndices> indices;
}physicalDevice;

class VkApplication
{
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

    VkPhysicalDevice                            & getPhysicalDevice();
    VkDevice                                    & getDevice();
    VkQueue                                     & getGraphicsQueue();
    std::vector<VkCommandPool>                  & getCommandPool();
    VkSurfaceKHR                                & getSurface();
    GLFWwindow                                  & getWindow();
    QueueFamilyIndices                          & getQueueFamilyIndices();

private:

    VkInstance                                  instance;
    GLFWwindow                                  *window;
    VkSurfaceKHR                                surface;
    VkDebugUtilsMessengerEXT                    debugMessenger;

    std::vector<physicalDevice>                 physicalDevices;
    VkPhysicalDevice                            physicalDevice = VK_NULL_HANDLE;
    QueueFamilyIndices                          indices;
    VkDevice                                    device;
    VkQueue                                     graphicsQueue;
    VkQueue                                     presentQueue;

    graphics                                    Graphics;
    postProcessing                              PostProcessing;
    uint32_t                                    imageCount;
    VkSampleCountFlagBits                       msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    std::vector<VkCommandPool>                  commandPool;
    std::vector<std::vector<VkCommandBuffer>>   commandBuffers;

    std::vector<VkSemaphore>                    imageAvailableSemaphores;
    std::vector<VkSemaphore>                    renderFinishedSemaphores;
    std::vector<VkFence>                        inFlightFences;
    std::vector<VkFence>                        imagesInFlight;

    size_t                                      currentFrame = 0;
    size_t                                      currentBuffer = 0;
    bool                                        framebufferResized = false;

    bool                                        updateCmdLight = false;
    bool                                        updateCmdWorld = false;
    unsigned long long                          updatedLightFrames = 0;
    unsigned long long                          updatedWorldFrames = 0;


    std::vector<texture             *>          textures;
    std::vector<gltfModel           *>          gltfModel;
    std::vector<object              *>          object3D;
    std::vector<light<spotLight>    *>          lightSource;
    std::vector<light<pointLight>   *>          lightPoint;
    std::vector<group               *>          groups;

    camera                                      *cam;
    texture                                     *emptyTexture;
    texture                                     *emptyTextureW;
    cubeTexture                                 *skybox;
    object                                      *skyboxObject;

    uint32_t                                    shadowCount = 0;

    double                                      xMpos, yMpos;
    double                                      angx=0.0, angy=0.0;

    float                                       frameTime;
    float                                       fps = 60.0f;
    bool                                        animate = true;
    bool                                        fpsLock = false;

    uint32_t                                    controledGroup = 0;

    void initWindow();
        static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

    //=================================Initialization===========================================//
    void initVulkan();

        void createInstance();
            std::vector<const char*> getRequiredExtensions();
            bool checkValidationLayerSupport();

        void setupDebugMessenger();
            void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
                static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                    VkDebugUtilsMessageTypeFlagsEXT messageType,const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,void* pUserData);
            VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);

        void createSurface();

        void pickPhysicalDevice();
        void createLogicalDevice();

        void createCommandPool();

        void createTextures();
        void loadModel();

        void createObjects();

        void createLight();

        void createGraphics();

        void updateLight();
        void updateObjectsUniformBuffers();

        void createDescriptors();

        void createCommandBuffers();
        void createCommandBuffer(uint32_t number);
        void updateCommandBuffer(uint32_t number, uint32_t i);

        void createSyncObjects();

    //=================================DrawLoop===========================================//
    void mainLoop();
        void updateAnimations(bool animate);
        void drawFrame();
            void recreateSwapChain();
                void cleanupSwapChain();
            void updateCmd(uint32_t imageIndex);
                void updateUniformBuffer(uint32_t currentImage);            
        void mouseEvent();
        static void scrol(GLFWwindow* window, double xoffset, double yoffset);

    //=================================Cleanup===========================================//
    void cleanup();
        void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
};


#endif // VULKANCORE_H
