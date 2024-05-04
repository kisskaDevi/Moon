#ifndef WORKFLOW_H
#define WORKFLOW_H

#include <vulkan.h>
#include "attachments.h"
#include "buffer.h"

#include <filesystem>

class workbody{
public:
    std::filesystem::path           vertShaderPath;
    std::filesystem::path           fragShaderPath;

    VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
    VkPipeline                      Pipeline{VK_NULL_HANDLE};
    VkDescriptorSetLayout           DescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    DescriptorSets;

    virtual ~workbody(){};
    virtual void destroy(VkDevice device);
    virtual void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass) = 0;
    virtual void createDescriptorSetLayout(VkDevice device) = 0;
};

class workflow
{
protected:
    VkPhysicalDevice                physicalDevice{VK_NULL_HANDLE};
    VkDevice                        device{VK_NULL_HANDLE};
    std::filesystem::path           shadersPath;
    moon::utils::ImageInfo                       image;

    VkRenderPass                    renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>      framebuffers;
    std::vector<VkCommandBuffer>    commandBuffers;
public:
    virtual ~workflow(){};
    virtual void destroy();

    workflow& setShadersPath(const std::filesystem::path &path);
    workflow& setDeviceProp(VkPhysicalDevice physicalDevice, VkDevice device);
    workflow& setImageProp(moon::utils::ImageInfo* pInfo);

    virtual void create(moon::utils::AttachmentsDatabase& aDatabase) = 0;
    virtual void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) = 0;
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
