#ifndef GRAPHICSMANAGER_H
#define GRAPHICSMANAGER_H

#define VK_USE_PLATFORM_WIN32_KHR
#include <libs/vulkan/vulkan.h>
#include <libs/glm/glm.hpp>
#include "core/operations.h"

#include "graphics/graphicsInterface.h"

#include <vector>

class                                           object;
class                                           camera;
struct                                          gltfModel;
class                                           spotLight;

struct physicalDevice{
    VkPhysicalDevice                            device;
    std::vector<QueueFamilyIndices>             indices;
};

class graphicsManager
{
public:
    graphicsManager();
    void cleanup();

    uint32_t                                    getImageIndex();
    uint32_t                                    getCurrentFrame();

    void                                        deviceWaitIdle();

    void                                        createInstance();
    void                                        setupDebugMessenger();
    void                                        createSurface(GLFWwindow* window);
    void                                        pickPhysicalDevice();
    void                                        createLogicalDevice();
    void                                        createCommandPool();
    void                                        setGraphics(graphicsInterface* graphics);
    void                                        createGraphics(GLFWwindow* window);
    void                                        createCommandBuffers();
    void                                        createSyncObjects();
    VkResult                                    checkNextFrame();
    VkResult                                    drawFrame();

private:
    #ifdef NDEBUG
        const bool                              enableValidationLayers = false;
    #else
        const bool                              enableValidationLayers = true;
    #endif
    const size_t                                MAX_FRAMES_IN_FLIGHT = 3;
    const std::vector<const char*>              validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*>              deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkInstance                                  instance{VK_NULL_HANDLE};
    VkSurfaceKHR                                surface{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT                    debugMessenger{VK_NULL_HANDLE};

    std::vector<physicalDevice>                 physicalDevices;
    uint32_t                                    physicalDeviceNumber;
    uint32_t                                    indicesNumber;

    VkDevice                                    device{VK_NULL_HANDLE};
    VkQueue                                     graphicsQueue{VK_NULL_HANDLE};
    VkQueue                                     presentQueue{VK_NULL_HANDLE};

    graphicsInterface*                          graphics{nullptr};

    VkCommandPool                               commandPool{VK_NULL_HANDLE};

    std::vector<VkSemaphore>                    imageAvailableSemaphores;
    std::vector<VkSemaphore>                    renderFinishedSemaphores;
    std::vector<VkFence>                        inFlightFences;
    std::vector<VkFence>                        imagesInFlight;

    uint32_t                                    imageIndex{0};
    uint32_t                                    currentFrame{0};
    bool                                        framebufferResized{false};

    std::vector<const char*>                    getRequiredExtensions();
    bool                                        checkValidationLayerSupport();

    void                                        populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    static VKAPI_ATTR VkBool32 VKAPI_CALL       debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,VkDebugUtilsMessageTypeFlagsEXT messageType,const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,void* pUserData);
    VkResult                                    CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);

    void                                        DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
};

#endif // GRAPHICSMANAGER_H
