#ifndef IMGUIGRAPHICS_H
#define IMGUIGRAPHICS_H

#include <vulkan.h>
#include <vector>

#include "graphicsInterface.h"
#include "imguiLink.h"

struct GLFWwindow;

namespace moon::imguiGraphics {

class ImguiGraphics: public moon::graphicsManager::GraphicsInterface
{
private:
    GLFWwindow*         window{nullptr};
    VkInstance          instance{VK_NULL_HANDLE};
    uint32_t            imageCount{0};
    VkDescriptorPool    descriptorPool{VK_NULL_HANDLE};
    VkCommandPool       commandPool{VK_NULL_HANDLE};
    ImguiLink           Link;


    void setupImguiContext();
    void createDescriptorPool();
    void createCommandPool();
    void uploadFonts();
public:
    ImguiGraphics(GLFWwindow* window, VkInstance instance, uint32_t maxImageCount);
    ~ImguiGraphics();

    void create() override;
    void destroy() override;
    void update(uint32_t imageIndex) override;
    std::vector<std::vector<VkSemaphore>> submit(const std::vector<std::vector<VkSemaphore>>& externalSemaphore, const std::vector<VkFence>& externalFence, uint32_t imageIndex) override;
};

}
#endif // IMGUIGRAPHICS_H
