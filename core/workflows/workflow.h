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
    moon::utils::ImageInfo          imageInfo;

    VkDevice device{VK_NULL_HANDLE};

    utils::vkDefault::PipelineLayout        pipelineLayout;
    utils::vkDefault::Pipeline              pipeline;
    utils::vkDefault::DescriptorSetLayout   descriptorSetLayout;

    utils::vkDefault::DescriptorPool descriptorPool;
    utils::vkDefault::DescriptorSets descriptorSets;

    virtual ~Workbody(){};
    Workbody() = default;
    Workbody(const moon::utils::ImageInfo& imageInfo) : imageInfo(imageInfo) {};
    Workbody(Workbody&&) = default;
    Workbody& operator=(Workbody&&) = default;

    virtual void create(
        const std::filesystem::path& vertShaderPath,
        const std::filesystem::path& fragShaderPath,
        VkDevice device,
        VkRenderPass pRenderPass) = 0;
};

class Workflow
{
protected:
    VkPhysicalDevice                physicalDevice{VK_NULL_HANDLE};
    VkDevice                        device{VK_NULL_HANDLE};
    std::filesystem::path           shadersPath;
    moon::utils::ImageInfo          imageInfo;

    utils::vkDefault::RenderPass    renderPass;
    utils::vkDefault::Framebuffers  framebuffers;
    std::vector<VkCommandBuffer>    commandBuffers;
public:
    virtual ~Workflow(){};
    Workflow(const moon::utils::ImageInfo& imageInfo, const std::filesystem::path & shadersPath) : imageInfo(imageInfo), shadersPath(shadersPath) {};
    Workflow(Workflow&&) = default;
    Workflow& operator=(Workflow&&) = default;

    Workflow& setDeviceProp(VkPhysicalDevice physicalDevice, VkDevice device);

    virtual void create(moon::utils::AttachmentsDatabase& aDatabase) = 0;
    virtual void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) = 0;
    virtual void updateCommandBuffer(uint32_t frameNumber) = 0;

    void createCommandBuffers(VkCommandPool commandPool);
    void beginCommandBuffer(uint32_t frameNumber);
    void endCommandBuffer(uint32_t frameNumber);
    VkCommandBuffer& getCommandBuffer(uint32_t frameNumber);
    void freeCommandBuffer(VkCommandPool commandPool);
};

}
#endif // WORKFLOW_H
