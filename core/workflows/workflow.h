#ifndef WORKFLOW_H
#define WORKFLOW_H

#include <vulkan.h>
#include "attachments.h"
#include "buffer.h"
#include "vkdefault.h"

#include <filesystem>
#include <unordered_map>
#include <memory>

namespace moon::workflows {

class Workbody{
public:
    std::filesystem::path   vertShaderPath;
    std::filesystem::path   fragShaderPath;
    moon::utils::ImageInfo  imageInfo;

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
    VkPhysicalDevice        physicalDevice{VK_NULL_HANDLE};
    VkDevice                device{VK_NULL_HANDLE};
    std::filesystem::path   shadersPath;
    moon::utils::ImageInfo  imageInfo;
    bool created{false};

    utils::vkDefault::RenderPass        renderPass;
    utils::vkDefault::Framebuffers      framebuffers;
    utils::vkDefault::CommandBuffers    commandBuffers;

public:
    virtual ~Workflow(){};
    Workflow(const moon::utils::ImageInfo& imageInfo, const std::filesystem::path & shadersPath) : imageInfo(imageInfo), shadersPath(shadersPath) {};
    Workflow(Workflow&&) = default;
    Workflow& operator=(Workflow&&) = default;

    Workflow& setDeviceProp(VkPhysicalDevice physicalDevice, VkDevice device);

    virtual void create(moon::utils::AttachmentsDatabase& aDatabase) = 0;
    virtual void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) = 0;
    virtual void updateCommandBuffer(uint32_t frameNumber) = 0;

    void createCommandBuffers(const utils::vkDefault::CommandPool& commandPool);
    void beginCommandBuffer(uint32_t frameNumber) const;
    void endCommandBuffer(uint32_t frameNumber) const;
    utils::vkDefault::CommandBuffer& commandBuffer(uint32_t frameNumber);
    void raiseUpdateFlags();
};

struct Parameters {
    bool enable{false};
    moon::utils::ImageInfo imageInfo;
    std::filesystem::path shadersPath;
};

using ParametersMap = std::unordered_map<std::string, moon::workflows::Parameters*>;
using WorkflowsMap = std::unordered_map<std::string, std::unique_ptr<moon::workflows::Workflow>>;

}
#endif // WORKFLOW_H
