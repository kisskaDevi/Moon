#ifndef IMGUIGRAPHICS_H
#define IMGUIGRAPHICS_H

#include <vulkan.h>
#include <vector>

#include "graphicsInterface.h"
#include "imguiLink.h"

struct GLFWwindow;
struct physicalDevice;
class swapChain;
class linkable;

class imguiGraphics: public graphicsInterface
{
private:
    VkInstance          instance{VK_NULL_HANDLE};
    VkDescriptorPool    descriptorPool{VK_NULL_HANDLE};
    VkCommandPool       commandPool{VK_NULL_HANDLE};
    imguiLink           Link;

    void setupImguiContext();
    void createDescriptorPool();
    void createCommandPool();
    void uploadFonts();
public:
    imguiGraphics();
    ~imguiGraphics();
    void destroyGraphics() override;

    void setInstance(VkInstance instance);
    void createGraphics() override;

    void updateCommandBuffer(uint32_t imageIndex) override;
    void updateBuffers(uint32_t imageIndex) override;

    std::vector<std::vector<VkSemaphore>> submit(const std::vector<std::vector<VkSemaphore>>& externalSemaphore, const std::vector<VkFence>& externalFence, uint32_t imageIndex) override;
};

#endif // IMGUIGRAPHICS_H
