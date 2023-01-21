#ifndef SHADOWGRAPHICS_H
#define SHADOWGRAPHICS_H

#include <libs/vulkan/vulkan.h>
#include "../attachments.h"

#include <vector>
#include <string>

struct Node;
class object;
struct QueueFamilyIndices;

class shadowGraphics
{
private:
    VkPhysicalDevice*                   physicalDevice{nullptr};
    VkDevice*                           device{nullptr};
    QueueFamilyIndices*                 queueFamilyIndices{nullptr};

    imageInfo                           image;

    attachment                          depthAttachment;

    VkRenderPass                        RenderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>          shadowMapFramebuffer;

    struct Shadow{
        std::string                     ExternalPath;

        VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
        VkPipeline                      Pipeline{VK_NULL_HANDLE};
        VkDescriptorSetLayout           DescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout           uniformBufferSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout           uniformBlockSetLayout{VK_NULL_HANDLE};
        VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>    DescriptorSets;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
    }shadow;

    VkCommandPool                       shadowCommandPool{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer>        shadowCommandBuffer;
    std::vector<bool>                   updateCommandBufferFlag;

    void renderNode(VkCommandBuffer commandBuffer, Node* node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets);
public:
    shadowGraphics(uint32_t imageCount, VkExtent2D shadowExtent = {1024,1024});
    void destroy();

    void setExternalPath(const std::string& path);
    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, QueueFamilyIndices* queueFamilyIndices);

    void createAttachments();

    void createCommandPool();

    void createRenderPass();
    void createFramebuffer();

    void createDescriptorPool();
    void createDescriptorSets();
    void updateDescriptorSets(uint32_t lightUniformBuffersCount, VkBuffer* plightUniformBuffers, unsigned long long sizeOfLightUniformBuffers);

    void createCommandBuffers();
    void updateCommandBuffer(uint32_t frameNumber, std::vector<object*>& objects);

    void createShadow();

    attachment*         getAttachment();
    VkCommandBuffer*    getCommandBuffer(uint32_t i);
};

#endif // SHADOWGRAPHICS_H
