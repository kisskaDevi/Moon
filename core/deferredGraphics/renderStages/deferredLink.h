#ifndef DEFERRED_LINK_H
#define DEFERRED_LINK_H

#include <vector>
#include <filesystem>

#include "linkable.h"
#include "attachments.h"
#include "vector.h"
#include "vkdefault.h"

namespace moon::deferredGraphics {

class Link : public moon::graphicsManager::Linkable
{
private:
    utils::vkDefault::PipelineLayout        pipelineLayout;
    utils::vkDefault::Pipeline              pipeline;
    utils::vkDefault::DescriptorSetLayout   descriptorSetLayout;

    utils::vkDefault::DescriptorPool        descriptorPool;
    utils::vkDefault::DescriptorSets        descriptorSets;

    struct PushConstant {
        moon::math::Vector<float, 2> offset{ 0.0f, 0.0f };
        moon::math::Vector<float, 2> size{ 1.0f, 1.0f };
    } pushConstant;

    void createDescriptorSetLayout(VkDevice device);
    void createPipeline(VkDevice device, const std::filesystem::path& shadersPath, const moon::utils::ImageInfo& info);
    void createDescriptors(VkDevice device, const moon::utils::ImageInfo& info, const moon::utils::Attachments* attachment);

public:
    Link() = default;
    Link(VkDevice device, const std::filesystem::path& shadersPath, const moon::utils::ImageInfo& info, VkRenderPass renderPass, const moon::utils::Attachments* attachment);

    void setPositionInWindow(const moon::math::Vector<float,2>& offset, const moon::math::Vector<float,2>& size) override;
    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
};

}
#endif // DEFERRED_LINK_H
