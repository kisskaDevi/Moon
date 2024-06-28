#ifndef NODE_H
#define NODE_H

#include <vulkan.h>
#include <vector>
#include <vkdefault.h>

namespace moon::utils {

struct Stage{
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkSemaphore> signalSemaphores;
    VkPipelineStageFlags waitStage;
    VkQueue queue{VK_NULL_HANDLE};
    VkFence fence{VK_NULL_HANDLE};

    Stage(std::vector<VkCommandBuffer> commandBuffers, VkPipelineStageFlags waitStages, VkQueue queue);

    VkResult submit();
};

struct Node{
    std::vector<Stage> stages;
    utils::vkDefault::Semaphores signalSemaphores;
    Node* next{nullptr};
    VkDevice device{VK_NULL_HANDLE};

    Node() = default;
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    void swap(Node&);
    Node(Node&&);
    Node& operator=(Node&&);
    Node(VkDevice device, const std::vector<Stage>& stages, Node* next);
    ~Node();

    Node* back();

    void setExternalSemaphore(const std::vector<std::vector<VkSemaphore>>& externalSemaphore);
    void setExternalFence(const std::vector<VkFence>& externalFence);
    std::vector<std::vector<VkSemaphore>> getBackSemaphores();

    VkResult createSemaphores();
    void submit();
};

}
#endif // NODE_H
