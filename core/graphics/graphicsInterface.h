#ifndef GRAPHICSINTERFACE_H
#define GRAPHICSINTERFACE_H

#include "libs/vulkan/vulkan.h"
#include <vector>
#include <optional>

class GLFWwindow;

struct deviceInfo
{
    VkPhysicalDevice*               physicalDevice;
    std::optional<uint32_t>*        graphicsFamily;
    std::optional<uint32_t>*        presentFamily;
    VkDevice*                       device;
    VkQueue*                        queue;
    VkCommandPool*                  commandPool;
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
    virtual void updateAllCommandBuffers() = 0;
    virtual void updateCommandBuffers(uint32_t imageIndex) = 0;
    virtual void freeCommandBuffers() = 0;

    virtual void updateBuffers(uint32_t imageIndex) = 0;

    virtual VkCommandBuffer*    getCommandBuffers(uint32_t& commandBuffersCount, uint32_t imageIndex) = 0;
    virtual uint32_t            getImageCount() = 0;
    virtual VkSwapchainKHR&     getSwapChain() = 0;
};

#endif // GRAPHICSINTERFACE_H
