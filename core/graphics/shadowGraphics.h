#ifndef SHADOWGRAPHICS_H
#define SHADOWGRAPHICS_H

#include <libs/vulkan/vulkan.h>
#include "attachments.h"

#include <vector>
#include <string>

struct Node;
class object;

struct shadowInfo{
    uint32_t                    imageCount;
    VkExtent2D                  extent;
    VkSampleCountFlagBits       msaaSamples;
    VkRenderPass                renderPass;
};

struct QueueFamilyIndices;

class shadowGraphics
{
private:
    VkPhysicalDevice*                   physicalDevice;
    VkDevice*                           device;
    QueueFamilyIndices*                 queueFamilyIndices;

    imageInfo                           image;

    attachment                          depthAttachment;
    VkSampler                           shadowSampler;

    VkRenderPass                        RenderPass;
    std::vector<VkFramebuffer>          shadowMapFramebuffer;

    struct Shadow{
        std::string                     ExternalPath;

        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;
        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorSetLayout           uniformBufferSetLayout;
        VkDescriptorSetLayout           uniformBlockSetLayout;
        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, shadowInfo info);
        void createDescriptorSetLayout(VkDevice* device);
    }shadow;

    VkCommandPool                       shadowCommandPool;
    std::vector<VkCommandBuffer>        shadowCommandBuffer;

    void renderNode(VkCommandBuffer commandBuffer, Node* node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets);
public:
    shadowGraphics(uint32_t imageCount, VkExtent2D shadowExtent = {1024,1024});
    void destroy();

    void setExternalPath(const std::string& path);
    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, QueueFamilyIndices* queueFamilyIndices);

    void createMap();
    void createMapView();
    void createSampler();

    void createCommandPool();

    void createRenderPass();
    void createFramebuffer();

    void createDescriptorPool();
    void createDescriptorSets();
    void updateDescriptorSets(uint32_t lightUniformBuffersCount, VkBuffer* plightUniformBuffers, unsigned long long sizeOfLightUniformBuffers);

    void createCommandBuffers();
    void updateCommandBuffer(uint32_t frameNumber, std::vector<object*>& objects);

    void createShadow();

    VkImageView                     & getImageView();
    VkSampler                       & getSampler();
    std::vector<VkCommandBuffer>    & getCommandBuffer();

    uint32_t                        getWidth() const;
    uint32_t                        getHeight() const;
};

#endif // SHADOWGRAPHICS_H
