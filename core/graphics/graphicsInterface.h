#ifndef GRAPHICSINTERFACE_H
#define GRAPHICSINTERFACE_H

#include "libs/vulkan/vulkan.h"
#include <vector>
#include <optional>

class GLFWwindow;

struct deviceInfo
{
    VkPhysicalDevice*               physicalDevice{VK_NULL_HANDLE};
    std::optional<uint32_t>*        graphicsFamily{nullptr};
    std::optional<uint32_t>*        presentFamily{nullptr};
    VkDevice*                       device{VK_NULL_HANDLE};
    VkQueue*                        queue{VK_NULL_HANDLE};
    VkCommandPool*                  commandPool{VK_NULL_HANDLE};
};

class graphicsInterface{
public:
    virtual ~graphicsInterface(){};
    virtual void destroyGraphics() = 0;

    virtual void setDevicesInfo(uint32_t devicesInfoCount, deviceInfo* devicesInfo) = 0;
    virtual void setSupportImageCount(VkSurfaceKHR* surface) = 0;
    virtual void createGraphics(GLFWwindow* window, VkSurfaceKHR* surface) = 0;
    virtual void updateDescriptorSets() = 0;

    virtual void createCommandBuffers() = 0;
    virtual void updateCommandBuffers() = 0;
    virtual void updateCommandBuffer(uint32_t imageIndex) = 0;
    virtual void freeCommandBuffers() = 0;

    virtual void updateBuffers(uint32_t imageIndex) = 0;

    virtual VkCommandBuffer*    getCommandBuffers(uint32_t& commandBuffersCount, uint32_t imageIndex) = 0;
    virtual uint32_t            getImageCount() = 0;
    virtual VkSwapchainKHR&     getSwapChain() = 0;
};

#endif // GRAPHICSINTERFACE_H
