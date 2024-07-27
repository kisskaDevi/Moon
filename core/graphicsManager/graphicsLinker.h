#ifndef GRAPHICSLINKER_H
#define GRAPHICSLINKER_H

#include <vector>
#include <stdint.h>
#include <vulkan.h>

#include "vkdefault.h"
#include "swapChain.h"
#include "linkable.h"
#include "graphicsInterface.h"

namespace moon::graphicsManager {

class GraphicsLinker
{
private:
    std::vector<GraphicsInterface*>* graphics;
    moon::utils::ImageInfo           imageInfo;

    utils::vkDefault::RenderPass     renderPass;
    utils::vkDefault::Framebuffers   framebuffers;
    utils::vkDefault::CommandPool    commandPool;
    utils::vkDefault::CommandBuffers commandBuffers;
    utils::vkDefault::Semaphores     signalSemaphores;

    void createRenderPass(VkDevice device);
    void createFramebuffers(VkDevice device, const moon::utils::SwapChain* swapChainKHR);
    void createCommandBuffers(VkDevice device);
    void createSyncObjects(VkDevice device);

public:
    GraphicsLinker() = default;
    GraphicsLinker(const GraphicsLinker&) = delete;
    GraphicsLinker& operator=(const GraphicsLinker&) = delete;
    GraphicsLinker(GraphicsLinker&&);
    GraphicsLinker& operator=(GraphicsLinker&&);
    void swap(GraphicsLinker&);

    GraphicsLinker(VkDevice device, const moon::utils::SwapChain* swapChainKHR, std::vector<GraphicsInterface*>* graphics);
    void update(uint32_t resourceNumber, uint32_t imageNumber);
    VkRenderPass getRenderPass() const;

    const VkSemaphore& submit(uint32_t frameNumber, const utils::vkDefault::VkSemaphores& waitSemaphores, VkFence fence, VkQueue queue);
};

}
#endif // GRAPHICSLINKER_H
