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

    virtual void createGraphics(uint32_t& imageCount, GLFWwindow* window, VkSurfaceKHR surface, VkExtent2D extent, VkSampleCountFlagBits MSAASamples, uint32_t devicesInfoCount, deviceInfo* devicesInfo) = 0;
    virtual void updateDescriptorSets() = 0;
    virtual void updateCommandBuffers(uint32_t imageCount, VkCommandBuffer* commandBuffers) = 0;
    virtual void fillCommandbufferSet(std::vector<VkCommandBuffer>& commandbufferSet, uint32_t imageIndex) = 0;
    virtual void updateCmd(uint32_t imageIndex, VkCommandBuffer* commandBuffers) = 0;
    virtual void updateUbo(uint32_t imageIndex) = 0;

    virtual VkSwapchainKHR& getSwapChain() = 0;
};

#endif // GRAPHICSINTERFACE_H
