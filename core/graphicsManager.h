#ifndef GRAPHICSMANAGER_H
#define GRAPHICSMANAGER_H

#define VK_USE_PLATFORM_WIN32_KHR
#include <libs/vulkan/vulkan.h>
#include "core/operations.h"

#include "graphics/graphicsInterface.h"

//#define NDEBUG

struct physicalDevice{
    VkPhysicalDevice                            device;
    std::vector<uint32_t>                       indices;
};

class graphicsManager
{
public:
    graphicsManager();
    void destroy();

    uint32_t                                    getImageIndex();
    void                                        deviceWaitIdle();

    void                                        createInstance();
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
        bool                                    enableValidationLayers = false;
    #else
        bool                                    enableValidationLayers = true;
    #endif

    const std::vector<const char*>              validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*>              deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkInstance                                  instance{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT                    debugMessenger{VK_NULL_HANDLE};
    VkSurfaceKHR                                surface{VK_NULL_HANDLE};

    std::vector<physicalDevice>                 physicalDevices;
    uint32_t                                    physicalDeviceNumber;
    uint32_t                                    indicesNumber;

    VkDevice                                    device{VK_NULL_HANDLE};
    VkQueue                                     graphicsQueue{VK_NULL_HANDLE};
    VkQueue                                     presentQueue{VK_NULL_HANDLE};

    graphicsInterface*                          graphics{nullptr};

    VkCommandPool                               commandPool{VK_NULL_HANDLE};

    VkSemaphore                                 availableSemaphores;
    std::vector<VkSemaphore>                    signalSemaphores;
    std::vector<VkFence>                        fences;

    uint32_t                                    imageIndex{0};
    bool                                        framebufferResized{false};
};

#endif // GRAPHICSMANAGER_H
