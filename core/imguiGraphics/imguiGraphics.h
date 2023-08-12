#ifndef IMGUIGRAPHICS_H
#define IMGUIGRAPHICS_H

#include <vulkan.h>
#include <vector>

#include "graphicsInterface.h"
#include "device.h"
#include "imguiLink.h"

struct GLFWwindow;
struct physicalDevice;
class swapChain;
class linkable;

class imguiGraphics: public graphicsInterface
{
private:
    uint32_t                                    imageCount{0};
    swapChain*                                  swapChainKHR{nullptr};

    std::vector<physicalDevice>                 devices;
    physicalDevice                              device;

    VkInstance                                  instance{VK_NULL_HANDLE};
    VkDescriptorPool                            descriptorPool{VK_NULL_HANDLE};
    VkCommandPool                               commandPool{VK_NULL_HANDLE};

    imguiLink                                   Link;
public:
    imguiGraphics() = default;
    ~imguiGraphics();
    void destroyGraphics() override;

    void setInstance(VkInstance instance);
    void setDevices(uint32_t devicesCount, physicalDevice* devices) override;
    void setSwapChain(swapChain* swapChainKHR) override;
    void createGraphics(GLFWwindow* window, VkSurfaceKHR surface) override;

    void setupImguiContext();
    void createDescriptorPool();
    void createCommandPool();
    void uploadFonts();

    void updateCommandBuffer(uint32_t imageIndex) override;
    void updateBuffers(uint32_t imageIndex) override;

    std::vector<std::vector<VkSemaphore>> sibmit(const std::vector<std::vector<VkSemaphore>>& externalSemaphore, const std::vector<VkFence>& externalFence, uint32_t imageIndex) override;

    linkable* getLinkable() override;
};

#endif // IMGUIGRAPHICS_H
