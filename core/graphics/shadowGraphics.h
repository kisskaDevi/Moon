#ifndef SHADOWGRAPHICS_H
#define SHADOWGRAPHICS_H

#include "core/vulkanCore.h"

struct shadowInfo{
    uint32_t                    imageCount;
    VkExtent2D                  extent;
    VkSampleCountFlagBits       msaaSamples;
    VkRenderPass                renderPass;
};

struct LightBufferObject
{
    alignas(16) glm::mat4   proj;
    alignas(16) glm::mat4   view;
    alignas(16) glm::mat4   projView;
    alignas(16) glm::vec4   position;
    alignas(16) glm::vec4   lightColor;
    alignas(4)  uint32_t    type;
};

class shadowGraphics
{
private:
    VkApplication                       *app;
    imageInfo                           image;

    attachment                          depthAttachment;
    VkSampler                           shadowSampler;

    VkRenderPass                        RenderPass;
    std::vector<VkFramebuffer>          shadowMapFramebuffer;

    struct Shadow{
        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;
        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorSetLayout           uniformBufferSetLayout;
        VkDescriptorSetLayout           uniformBlockSetLayout;
        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;

        void Destroy(VkApplication  *app);
        void createPipeline(VkApplication *app, shadowInfo info);
        void createDescriptorSetLayout(VkApplication *app);
    }shadow;

    VkCommandPool                       shadowCommandPool;
    std::vector<VkCommandBuffer>        shadowCommandBuffer;

    void renderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets);
public:
    shadowGraphics(VkApplication *app, uint32_t imageCount, VkExtent2D shadowExtent = {1024,1024});
    void destroy();

    void createMap();
    void createMapView();
    void createSampler();

    void createCommandPool();

    void createRenderPass();
    void createFramebuffer();

    void createDescriptorPool();
    void createDescriptorSets();
    void updateDescriptorSets(uint32_t lightUniformBuffersCount, VkBuffer* plightUniformBuffers);

    void createCommandBuffers();
    void updateCommandBuffer(uint32_t frameNumber, ShadowPassObjects objects);

    void createShadow();

    VkImageView                     & getImageView();
    VkSampler                       & getSampler();
    std::vector<VkCommandBuffer>    & getCommandBuffer();

    uint32_t                        getWidth() const;
    uint32_t                        getHeight() const;
};

#endif // SHADOWGRAPHICS_H
