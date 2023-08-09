#ifndef WORKFLOW_H
#define WORKFLOW_H

#include <vulkan.h>
#include "attachments.h"

#include <filesystem>

class texture;

class workbody{
public:
    std::filesystem::path           vertShaderPath;
    std::filesystem::path           fragShaderPath;

    VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
    VkPipeline                      Pipeline{VK_NULL_HANDLE};
    VkDescriptorSetLayout           DescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    DescriptorSets;

    void destroy(VkDevice device);
    virtual void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) = 0;
    virtual void createDescriptorSetLayout(VkDevice device) = 0;
};

class workflow
{
protected:
    VkPhysicalDevice                physicalDevice{VK_NULL_HANDLE};
    VkDevice                        device{VK_NULL_HANDLE};
    std::filesystem::path           shadersPath;
    imageInfo                       image;
    texture*                        emptyTexture;

    uint32_t                        attachmentsCount{0};
    attachments*                    pAttachments{nullptr};

    VkRenderPass                    renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>      framebuffers;
    std::vector<VkCommandBuffer>    commandBuffers;
public:
    virtual ~workflow(){};
    void destroy();

    void setEmptyTexture(texture* emptyTexture);
    void setShadersPath(const std::filesystem::path &path);
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

    static void createDescriptorPool(VkDevice device, workbody* workbody, const uint32_t& bufferCount, const uint32_t& imageCount, const uint32_t& maxSets);
    static void createDescriptorSets(VkDevice device, workbody* workbody, const uint32_t& imageCount);
};

#endif // WORKFLOW_H
