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

    virtual void createGraphics(GLFWwindow* window, VkSurfaceKHR* surface, uint32_t devicesInfoCount, deviceInfo* devicesInfo) = 0;
    virtual void updateDescriptorSets() = 0;
    virtual void updateCommandBuffers(VkCommandBuffer* commandBuffers) = 0;
    virtual void fillCommandBufferSet(std::vector<VkCommandBuffer>& commandbufferSet, uint32_t imageIndex) = 0;
    virtual void updateCmd(uint32_t imageIndex, VkCommandBuffer* commandBuffers) = 0;
    virtual void updateUbo(uint32_t imageIndex) = 0;

    virtual uint32_t        getImageCount() = 0;
    virtual VkSwapchainKHR& getSwapChain() = 0;
};

#endif // GRAPHICSINTERFACE_H
