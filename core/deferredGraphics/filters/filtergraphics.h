#ifndef FILTERGRAPHICS_H
#define FILTERGRAPHICS_H

#include <vulkan.h>
#include "../../utils/attachments.h"

#include <string>

class texture;

class filter{
public:
    std::string                     vertShaderPath;
    std::string                     fragShaderPath;

    VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
    VkPipeline                      Pipeline{VK_NULL_HANDLE};
    VkDescriptorSetLayout           DescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    DescriptorSets;

    void destroy(VkDevice device);
    virtual void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) = 0;
    virtual void createDescriptorSetLayout(VkDevice device) = 0;
};

class filterGraphics
{
protected:
    VkPhysicalDevice    physicalDevice{VK_NULL_HANDLE};
    VkDevice            device{VK_NULL_HANDLE};
    std::string         externalPath{};
    imageInfo           image;
    texture*            emptyTexture;

    uint32_t            attachmentsCount{0};
    attachments*        pAttachments{nullptr};

    VkRenderPass                    renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>      framebuffers;
    std::vector<VkCommandBuffer>    commandBuffers;
public:
    virtual ~filterGraphics(){};
    void destroy();

    void setEmptyTexture(texture* emptyTexture);
    void setExternalPath(const std::string &path);
    void setDeviceProp(VkPhysicalDevice physicalDevice, VkDevice device);
    void setImageProp(imageInfo* pInfo);
    void setAttachments(uint32_t attachmentsCount, attachments* pAttachments);

    virtual void createRenderPass() = 0;
    virtual void createFramebuffers() = 0;
    virtual void createPipelines() = 0;

    virtual void createDescriptorPool() = 0;
    virtual void createDescriptorSets() = 0;

    virtual void updateCommandBuffer(uint32_t frameNumber) = 0;

    void createCommandBuffers(VkCommandPool commandPool);
    void beginCommandBuffer(uint32_t frameNumber);
    void endCommandBuffer(uint32_t frameNumber);
    VkCommandBuffer& getCommandBuffer(uint32_t frameNumber);
    void freeCommandBuffer(VkCommandPool commandPool);

    static void createDescriptorPool(VkDevice device, filter* filter, const uint32_t& bufferCount, const uint32_t& imageCount, const uint32_t& maxSets);
    static void createDescriptorSets(VkDevice device, filter* filter, const uint32_t& imageCount);
};

#endif // FILTERGRAPHICS_H
