#ifndef GRAPHICSMANAGER_H
#define GRAPHICSMANAGER_H

#ifdef WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <vulkan.h>
#include "operations.h"
#include "device.h"
#include "swapChain.h"

#include "graphicsInterface.h"
#include "graphicsLinker.h"

//#define NDEBUG

namespace moon::graphicsManager {

class GraphicsManager
{
public:
    GraphicsManager(GLFWwindow* window, int32_t imageCount = -1, int32_t resourceCount = -1, const VkPhysicalDeviceFeatures& deviceFeatures = {});
    ~GraphicsManager();

    VkInstance getInstance() const;
    VkExtent2D getImageExtent() const;
    uint32_t   getResourceIndex() const;
    uint32_t   getResourceCount() const;
    uint32_t   getImageIndex() const;
    uint32_t   getImageCount() const;

    std::vector<moon::utils::PhysicalDeviceProperties> getDeviceInfo() const;
    void setDevice(uint32_t deviceIndex);
    void setGraphics(GraphicsInterface* graphics);

    void create(GLFWwindow* window);
    void destroy();

    VkResult checkNextFrame();
    VkResult drawFrame();
    VkResult deviceWaitIdle() const;

    std::vector<uint32_t> makeScreenshot() const;

private:
    bool                                        enableValidationLayers = true;

    const std::vector<const char*>              validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*>              deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    utils::vkDefault::Instance                  instance;
    utils::vkDefault::DebugUtilsMessenger       debugMessenger;
    utils::vkDefault::Surface                   surface;

    moon::utils::PhysicalDeviceMap              devices;
    moon::utils::PhysicalDevice*                activeDevice{nullptr};
    moon::utils::SwapChain                      swapChainKHR;

    std::vector<GraphicsInterface*>             graphics;
    GraphicsLinker                              linker;

    std::vector<VkSemaphore>                    availableSemaphores;
    std::vector<VkFence>                        fences;

    uint32_t                                    imageIndex{0};
    uint32_t                                    imageCount{0};
    uint32_t                                    resourceIndex{0};
    uint32_t                                    resourceCount{0};

    VkResult createDevice(const VkPhysicalDeviceFeatures& deviceFeatures = {});
    VkResult createInstance();
    VkResult createSurface(GLFWwindow* window);
    VkResult createSwapChain(GLFWwindow* window, int32_t maxImageCount = -1);
    VkResult createLinker();
    VkResult createSyncObjects();

    void destroyLinker();
    void destroySyncObjects();
};

}
#endif // GRAPHICSMANAGER_H
