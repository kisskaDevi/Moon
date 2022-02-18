#ifndef SHADOWGRAPHICS_H
#define SHADOWGRAPHICS_H

#include "core/vulkanCore.h"

const int MAX_LIGHT_SOURCE_COUNT = 16;

struct shadowInfo{
    uint32_t                    imageCount;
    uint32_t                    width;
    uint32_t                    height;
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
    alignas(4)  uint32_t    enableShadow;
};

struct LightUniformBufferObject
{
    LightBufferObject buffer[MAX_LIGHT_SOURCE_COUNT];
};

class shadowGraphics
{
private:
    VkApplication                       *app;

    struct Image{
        uint32_t                            Count;
        VkExtent2D                          Extent;
        uint32_t                            MipLevels = 1;
    }image;

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

    std::vector<VkCommandPool>                      shadowCommandPool;
    std::vector<std::vector<VkCommandBuffer>>       shadowCommandBuffer;

    void renderNode(Node *node, VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, VkDescriptorSet& objectDescriptorSet);
public:
    shadowGraphics(VkApplication *app, uint32_t imageCount);
    void destroy();

    void createMap();
    void createMapView();
    void createSampler();

    void createCommandPool(uint32_t commandPoolCount);

    void createRenderPass();
    void createFramebuffer();

    void createDescriptorPool();
    void createDescriptorSets(std::vector<VkBuffer> &lightUniformBuffers);

    void createCommandBuffers(uint32_t number);
    void updateCommandBuffers(uint32_t number, uint32_t i, std::vector<object *> & object3D, uint32_t lightNumber);

    void createShadow(uint32_t commandPoolsCount);

    VkImageView                     & getImageView();
    VkSampler                       & getSampler();
    std::vector<VkCommandBuffer>    & getCommandBuffer(uint32_t number);

    uint32_t                        getWidth() const;
    uint32_t                        getHeight() const;
};

#endif // SHADOWGRAPHICS_H
