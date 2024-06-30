#ifndef GRAPHICSLINKER_H
#define GRAPHICSLINKER_H

#include <vector>
#include <stdint.h>
#include <vulkan.h>

#include "vkdefault.h"
#include "swapChain.h"

namespace moon::graphicsManager {

class Linkable;

class GraphicsLinker
{
private:
    moon::utils::ImageInfo imageInfo;

    VkDevice                        device{ VK_NULL_HANDLE };
    std::vector<Linkable*>          linkables;
    moon::utils::SwapChain*         swapChainKHR{nullptr};

    utils::vkDefault::RenderPass    renderPass;
    utils::vkDefault::Framebuffers  framebuffers;

    utils::vkDefault::CommandPool   commandPool;
    std::vector<VkCommandBuffer>    commandBuffers;
    std::vector<bool>               updateCommandBufferFlags;

    utils::vkDefault::Semaphores    signalSemaphores;

public:
    ~GraphicsLinker();
    void destroy();

    void setSwapChain(moon::utils::SwapChain* swapChainKHR);
    void setDevice(VkDevice device);
    void addLinkable(Linkable* link);

    void createRenderPass();
    void createFramebuffers();

    void createCommandBuffers();
    void updateCommandBuffer(uint32_t resourceNumber, uint32_t imageNumber);

    void createSyncObjects();
    const VkSemaphore& submit(uint32_t frameNumber, const std::vector<VkSemaphore>& waitSemaphores, VkFence fence, VkQueue queue);

    const VkRenderPass& getRenderPass() const;
    const VkCommandBuffer& getCommandBuffer(uint32_t frameNumber) const;

    void updateCmdFlags();
};

}
#endif // GRAPHICSLINKER_H
