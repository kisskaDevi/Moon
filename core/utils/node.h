#ifndef NODE_H
#define NODE_H

#include <vector>

#include <vulkan.h>
#include <vkdefault.h>

namespace moon::utils {

struct PipelineStage{
    struct Frame {
        std::vector<VkCommandBuffer> commandBuffers;
        std::vector<VkSemaphore> wait;
        utils::vkDefault::Semaphores signal;
    };
    std::vector<Frame> frames;

    VkPipelineStageFlags waitStagesMask{};
    VkQueue queue{VK_NULL_HANDLE};
    VkFence fence{VK_NULL_HANDLE};

    PipelineStage(const std::vector<const vkDefault::CommandBuffers*>& commandBuffers, VkPipelineStageFlags waitStagesMask, VkQueue queue);
    VkResult submit(uint32_t frameIndex) const;
};

using PipelineStages = std::vector<PipelineStage>;

class PipelineNode{
private:
    utils::PipelineStages stages;
    PipelineNode* next{nullptr};

    std::vector<std::vector<VkSemaphore>> semaphores(uint32_t frameIndex);

public:
    PipelineNode() = default;
    PipelineNode(VkDevice device, PipelineStages&& stages, PipelineNode* next = nullptr);

    std::vector<std::vector<VkSemaphore>> submit(
        const uint32_t frameIndex,
        const std::vector<VkFence>& externalFence = {},
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore = {});
};

using PipelineNodes = std::vector<moon::utils::PipelineNode>;

}
#endif // NODE_H
