#ifndef GRAPHICSMANAGER_H
#define GRAPHICSMANAGER_H

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan.h>
#include "utils/operations.h"
#include "utils/device.h"

#include "graphicsInterface.h"

//#define NDEBUG

class graphicsManager
{
public:
    graphicsManager();
    void destroy();

    uint32_t                                    getImageIndex();
    void                                        deviceWaitIdle();

    void                                        createInstance();
    void                                        createSurface(GLFWwindow* window);
    void                                        createDevice();
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

    std::vector<physicalDevice>                 devices;

    graphicsInterface*                          graphics{nullptr};

    std::vector<VkSemaphore>                    availableSemaphores;
    std::vector<VkSemaphore>                    signalSemaphores;
    std::vector<VkFence>                        fences;

    uint32_t                                    imageIndex{0};
    uint32_t                                    semaphorIndex{0};
};

#endif // GRAPHICSMANAGER_H
