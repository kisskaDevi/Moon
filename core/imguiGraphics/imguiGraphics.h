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
    GLFWwindow*         window{nullptr};
    VkInstance          instance{VK_NULL_HANDLE};
    uint32_t            imageCount{0};
    VkDescriptorPool    descriptorPool{VK_NULL_HANDLE};
    VkCommandPool       commandPool{VK_NULL_HANDLE};
    imguiLink           Link;

    void setupImguiContext();
    void createDescriptorPool();
    void createCommandPool();
    void uploadFonts();
public:
    imguiGraphics(GLFWwindow* window, VkInstance instance, uint32_t maxImageCount);
    ~imguiGraphics();

    void create() override;
    void destroy() override;
    void update(uint32_t imageIndex) override;
    std::vector<std::vector<VkSemaphore>> submit(const std::vector<std::vector<VkSemaphore>>& externalSemaphore, const std::vector<VkFence>& externalFence, uint32_t imageIndex) override;
};

#endif // IMGUIGRAPHICS_H
