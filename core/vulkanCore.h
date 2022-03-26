#ifndef VULKANCORE_H
#define VULKANCORE_H

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3native.h>

class camera;
struct gltfModel;

template <typename type> class light;
class spotLight;
template <> class light<spotLight>;

#include <iostream>
#include <vector>
#include <set>

#include "graphics/graphics.h"

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

const int MAX_FRAMES_IN_FLIGHT = 3;
const int COMMAND_POOLS = 1;

const std::string ExternalPath = "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\";
const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

struct physicalDevice{
    VkPhysicalDevice                            device;
    std::vector<QueueFamilyIndices>             indices;
};

struct updateFlag{
    bool                                        enable = false;
    uint32_t                                    frames = 0;
};

class VkApplication
{
public:
    void initializeVulkan();
    void mainLoop(GLFWwindow* window, bool &framebufferResized);
    void cleanup();

    VkPhysicalDevice                            & getPhysicalDevice();
    VkDevice                                    & getDevice();
    VkQueue                                     & getGraphicsQueue();
    std::vector<VkCommandPool>                  & getCommandPool();
    VkSurfaceKHR                                & getSurface();
    QueueFamilyIndices                          & getQueueFamilyIndices();
    graphics                                    & getGraphics();

    void                                        resetCmdLight();
    void                                        resetCmdWorld();
    void                                        resetUboLight();
    void                                        resetUboWorld();

    void                                        addlightSource(light<spotLight>* lightSource);
    void                                        addCamera(camera * cameras);

//step 1
    void                                        createInstance();
    void                                        setupDebugMessenger();
    void                                        createSurface(GLFWwindow* window);
//step 2
    void                                        pickPhysicalDevice();
    void                                        createLogicalDevice();
    void                                        checkSwapChainSupport();
//step 3
    void                                        createCommandPool();
//step 4
    void                                        updateLight();
//step 5
    void                                        createGraphics(GLFWwindow* window);
    void                                        updateDescriptorSets();
//step 6
    void                                        createCommandBuffers();
    void                                        createSyncObjects();
//step 7
    VkResult                                    drawFrame();
    void                                        cleanupSwapChain();

private:

    VkInstance                                  instance;
    VkSurfaceKHR                                surface;
    VkDebugUtilsMessengerEXT                    debugMessenger;

    std::vector<physicalDevice>                 physicalDevices;
    uint32_t                                    physicalDeviceNumber;
    uint32_t                                    indicesNumber;

    VkDevice                                    device;
    VkQueue                                     graphicsQueue;
    VkQueue                                     presentQueue;

    SwapChainSupportDetails                     swapChainSupport;
    uint32_t                                    imageCount;

    graphics                                    Graphics;
    postProcessing                              PostProcessing;

    std::vector<VkCommandPool>                  commandPool;
    std::vector<std::vector<VkCommandBuffer>>   commandBuffers;

    std::vector<VkSemaphore>                    imageAvailableSemaphores;
    std::vector<VkSemaphore>                    renderFinishedSemaphores;
    std::vector<VkFence>                        inFlightFences;
    std::vector<VkFence>                        imagesInFlight;

    uint32_t                                    currentFrame = 0;
    uint32_t                                    currentBuffer = 0;
    bool                                        framebufferResized = false;

    updateFlag                                  worldCmd;
    updateFlag                                  lightsCmd;
    updateFlag                                  worldUbo;
    updateFlag                                  lightsUbo;

    std::vector<light<spotLight>    *>          lightSources;

    camera                                      *cameras;

    //=================================InitializeVulkan===========================================//

        std::vector<const char*>                getRequiredExtensions();
        bool                                    checkValidationLayerSupport();

        void                                    populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
        static VKAPI_ATTR VkBool32 VKAPI_CALL   debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,VkDebugUtilsMessageTypeFlagsEXT messageType,const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,void* pUserData);
        VkResult                                CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);

        void                                    createCommandBuffer(uint32_t number);
        void                                    updateCommandBuffer(uint32_t number, uint32_t i);

    //=================================DrawLoop===========================================//
        void                                    updateCmd(uint32_t imageIndex);
        void                                    updateUbo(uint32_t imageIndex);
        void                                    updateUniformBuffer(uint32_t currentImage);

    //=================================Cleanup===========================================//
        void                                    DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
};

#endif // VULKANCORE_H
