#ifndef GRAPHICSMANAGER_H
#define GRAPHICSMANAGER_H

#include <vulkan.h>
#include "operations.h"
#include "device.h"
#include "swapChain.h"

#include "graphicsInterface.h"
#include "graphicsLinker.h"

//#define NDEBUG

class graphicsManager
{
public:
    graphicsManager();
    void destroy();
    void destroySwapChain();
    void destroyLinker();

    VkInstance getInstance(){
        return instance;
    }

    VkSurfaceKHR    getSurface();
    uint32_t        getImageIndex();
    uint32_t        getImageCount();
    VkResult        deviceWaitIdle();

    VkResult        createInstance();
    VkResult        createSurface(GLFWwindow* window);
    VkResult        createDevice();
    VkResult        createSwapChain(GLFWwindow* window, int32_t maxImageCount = -1);
    VkResult        createLinker();
    void            setGraphics(graphicsInterface* graphics);
    VkResult        createSyncObjects();
    VkResult        checkNextFrame();
    VkResult        drawFrame();

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

    swapChain                                   swapChainKHR;

    std::vector<graphicsInterface*>             graphics;
    graphicsLinker                              linker;

    std::vector<VkSemaphore>                    availableSemaphores;
    std::vector<VkFence>                        fences;

    uint32_t                                    imageIndex{0};
    uint32_t                                    semaphorIndex{0};
};

#endif // GRAPHICSMANAGER_H
