#ifndef RAYTRACINGLINK
#define RAYTRACINGLINK

#include <vector>
#include <filesystem>

#include "linkable.h"
#include "attachments.h"
#include "vector.h"

namespace moon::rayTracingGraphics {

struct RayTracingLinkParameters{
    struct{
        std::string color;
        std::string bloom;
        std::string boundingBox;
    }in;
    struct{}out;
    std::filesystem::path shadersPath;
    moon::utils::ImageInfo imageInfo;
};

class RayTracingLink : public moon::graphicsManager::Linkable{
private:
    RayTracingLinkParameters                parameters;
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
    void createPipeline(VkDevice device);
    void createDescriptors(VkDevice device, const moon::utils::AttachmentsDatabase& aDatabase);

public:
    RayTracingLink() = default;
    RayTracingLink(VkDevice device, const RayTracingLinkParameters& parameters, VkRenderPass renderPass, const moon::utils::AttachmentsDatabase& aDatabase);

    void setPositionInWindow(const math::Vector<float, 2>& offset, const math::Vector<float, 2>& size) override;
    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
};

}
#endif
