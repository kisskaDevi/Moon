#ifndef NODE_H
#define NODE_H

#include <vulkan.h>
#include <vector>

namespace moon::utils {

struct Stage{
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkSemaphore> signalSemaphores;
    VkPipelineStageFlags waitStage;
    VkQueue queue{VK_NULL_HANDLE};
    VkFence fence{VK_NULL_HANDLE};

    Stage(  std::vector<VkCommandBuffer> commandBuffers,
            VkPipelineStageFlags waitStages,
            VkQueue queue);

    VkResult submit();
};

struct Node{
    std::vector<Stage> stages;
    std::vector<VkSemaphore> signalSemaphores;
    Node* next{nullptr};

    Node(const std::vector<Stage>& stages, Node* next);
    void destroy(VkDevice device);

    Node* back();

    void setExternalSemaphore(const std::vector<std::vector<VkSemaphore>>& externalSemaphore);
    void setExternalFence(const std::vector<VkFence>& externalFence);
    std::vector<std::vector<VkSemaphore>> getBackSemaphores();

    VkResult createSemaphores(VkDevice device);
    void submit();
};

}
#endif // NODE_H
