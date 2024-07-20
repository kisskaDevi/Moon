#ifndef NODE_H
#define NODE_H

#include <vulkan.h>
#include <vector>
#include <vkdefault.h>

namespace moon::utils {

struct PipelineStage{
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkSemaphore> signalSemaphores;
    VkPipelineStageFlags waitStage;
    VkQueue queue{VK_NULL_HANDLE};
    VkFence fence{VK_NULL_HANDLE};

    PipelineStage(std::vector<VkCommandBuffer> commandBuffers, VkPipelineStageFlags waitStages, VkQueue queue);

    VkResult submit();
};

using PipelineStages = std::vector<PipelineStage>;

struct PipelineNode{
    utils::PipelineStages stages;
    utils::vkDefault::Semaphores signalSemaphores;
    PipelineNode* next{nullptr};
    VkDevice device{VK_NULL_HANDLE};

    PipelineNode() = default;
    PipelineNode(const PipelineNode&) = delete;
    PipelineNode& operator=(const PipelineNode&) = delete;

    void swap(PipelineNode&);
    PipelineNode(PipelineNode&&);
    PipelineNode& operator=(PipelineNode&&);
    PipelineNode(VkDevice device, const PipelineStages& stages, PipelineNode* next = nullptr);
    ~PipelineNode();

    PipelineNode* back();

    void setExternalSemaphore(const std::vector<std::vector<VkSemaphore>>& externalSemaphore);
    void setExternalFence(const std::vector<VkFence>& externalFence);
    std::vector<std::vector<VkSemaphore>> getBackSemaphores();

    VkResult createSemaphores();
    void submit();
};

using PipelineNodes = std::vector<moon::utils::PipelineNode>;

}
#endif // NODE_H
