#ifndef NODE_H
#define NODE_H

#include <vulkan.h>
#include <vector>

struct stage
{
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkPipelineStageFlags> waitStages;
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkSemaphore> signalSemaphores;
    VkQueue queue{VK_NULL_HANDLE};
    VkFence fence{VK_NULL_HANDLE};

    stage(  std::vector<VkCommandBuffer> commandBuffers,
            std::vector<VkPipelineStageFlags> waitStages,
            VkQueue queue);

    VkResult submit();
};

struct node
{
    std::vector<stage> stages;
    std::vector<VkSemaphore> signalSemaphores;
    node* next{nullptr};

    node(const std::vector<stage>& stages, node* next);
    void destroy(VkDevice device);

    node* back();

    void setExternalSemaphore(const std::vector<std::vector<VkSemaphore>>& externalSemaphore);
    void setExternalFence(std::vector<VkFence>& externalFence);
    std::vector<std::vector<VkSemaphore>> getBackSemaphores();

    VkResult createSemaphores(VkDevice device);
    void submit();
};

#endif // NODE_H
