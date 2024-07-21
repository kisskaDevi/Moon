#ifndef DEFERRED_LINK_H
#define DEFERRED_LINK_H

#include <vector>
#include <filesystem>

#include "linkable.h"
#include "attachments.h"
#include "vector.h"
#include "vkdefault.h"

namespace moon::deferredGraphics {

class Link : public graphicsManager::Linkable
{
private:
    utils::vkDefault::PipelineLayout        pipelineLayout;
    utils::vkDefault::Pipeline              pipeline;
    utils::vkDefault::DescriptorSetLayout   descriptorSetLayout;

    utils::vkDefault::DescriptorPool        descriptorPool;
    utils::vkDefault::DescriptorSets        descriptorSets;

    void createDescriptorSetLayout(VkDevice device);
    void createPipeline(VkDevice device, const std::filesystem::path& shadersPath, const utils::ImageInfo& info);
    void createDescriptors(VkDevice device, const utils::ImageInfo& info, const utils::Attachments* attachment);

public:
    Link() = default;
    Link(VkDevice device, const std::filesystem::path& shadersPath, const utils::ImageInfo& info, VkRenderPass renderPass, const utils::Attachments* attachment);

    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
};

}
#endif // DEFERRED_LINK_H
