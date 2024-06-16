#ifndef WORKFLOW_H
#define WORKFLOW_H

#include <vulkan.h>
#include "attachments.h"
#include "buffer.h"
#include "vkdefault.h"

#include <filesystem>

namespace moon::workflows {

class Workbody{
public:
    std::filesystem::path           vertShaderPath;
    std::filesystem::path           fragShaderPath;

    utils::vkDefault::PipelineLayout        pipelineLayout;
    utils::vkDefault::Pipeline              pipeline;
    utils::vkDefault::DescriptorSetLayout   descriptorSetLayout;

    VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    DescriptorSets;

    virtual ~Workbody(){};
    Workbody() = default;
    Workbody(Workbody&&) = default;
    Workbody& operator=(Workbody&&) = default;
    virtual void destroy(VkDevice device);
    virtual void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass) = 0;
    virtual void createDescriptorSetLayout(VkDevice device) = 0;
};

class Workflow
{
protected:
    VkPhysicalDevice                physicalDevice{VK_NULL_HANDLE};
    VkDevice                        device{VK_NULL_HANDLE};
    std::filesystem::path           shadersPath;
    moon::utils::ImageInfo          image;

    utils::vkDefault::RenderPass    renderPass;
    std::vector<VkFramebuffer>      framebuffers;
    std::vector<VkCommandBuffer>    commandBuffers;
public:
    virtual ~Workflow(){};
    Workflow() = default;
    Workflow(Workflow&&) = default;
    Workflow& operator=(Workflow&&) = default;
    virtual void destroy();

    Workflow& setShadersPath(const std::filesystem::path &path);
    Workflow& setDeviceProp(VkPhysicalDevice physicalDevice, VkDevice device);
    Workflow& setImageProp(moon::utils::ImageInfo* pInfo);

    virtual void create(moon::utils::AttachmentsDatabase& aDatabase) = 0;
    virtual void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) = 0;
    virtual void updateCommandBuffer(uint32_t frameNumber) = 0;

    void createCommandBuffers(VkCommandPool commandPool);
    void beginCommandBuffer(uint32_t frameNumber);
    void endCommandBuffer(uint32_t frameNumber);
    VkCommandBuffer& getCommandBuffer(uint32_t frameNumber);
    void freeCommandBuffer(VkCommandPool commandPool);

    static void createDescriptorPool(VkDevice device, Workbody* workbody, const uint32_t& bufferCount, const uint32_t& imageCount, const uint32_t& maxSets);
    static void createDescriptorSets(VkDevice device, Workbody* workbody, const uint32_t& imageCount);
};

}
#endif // WORKFLOW_H
