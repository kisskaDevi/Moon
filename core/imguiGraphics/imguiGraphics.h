#ifndef IMGUIGRAPHICS_H
#define IMGUIGRAPHICS_H

#include <vulkan.h>
#include <vector>

#include "vkdefault.h"
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

    utils::vkDefault::DescriptorPool descriptorPool;
    utils::vkDefault::CommandPool commandPool;

    void setupImguiContext();
    void uploadFonts();

    void update(uint32_t imageIndex) override;
    utils::vkDefault::VkSemaphores submit(uint32_t frameIndex, const std::vector<VkFence>& externalFence = {}, const utils::vkDefault::VkSemaphores& externalSemaphore = {}) override;
public:
    ImguiGraphics(GLFWwindow* window, VkInstance instance, uint32_t maxImageCount);
    ~ImguiGraphics();

    void reset() override;
};

}
#endif // IMGUIGRAPHICS_H
