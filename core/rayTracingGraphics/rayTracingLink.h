#ifndef RAYTRACINGLINK
#define RAYTRACINGLINK

#include <vector>
#include <filesystem>

#include "linkable.h"
#include "attachments.h"
#include "vector.h"

namespace moon::rayTracingGraphics {

struct LinkPushConstant{
    moon::math::Vector<float,2> offset{0.0f, 0.0f};
    moon::math::Vector<float,2> size{1.0f, 1.0f};
};

struct RayTracingLinkParameters{
    struct{
        std::string color;
        std::string bloom;
        std::string boundingBox;
    }in;
    struct{}out;
};

class RayTracingLink : public moon::graphicsManager::Linkable{
private:
    RayTracingLinkParameters        parameters;
    std::filesystem::path           shadersPath;
    moon::utils::ImageInfo          imageInfo;

    VkDevice        device{VK_NULL_HANDLE};
    VkRenderPass    renderPass{VK_NULL_HANDLE};

    utils::vkDefault::PipelineLayout        pipelineLayout;
    utils::vkDefault::Pipeline              pipeline;
    utils::vkDefault::DescriptorSetLayout   descriptorSetLayout;
    utils::vkDefault::DescriptorPool        descriptorPool;
    utils::vkDefault::DescriptorSets        descriptorSets;

    LinkPushConstant pushConstant;

    void createDescriptorSetLayout();
    void createPipeline();
    void createDescriptors();

public:
    RayTracingLink() = default;
    RayTracingLink(const RayTracingLinkParameters& parameters) : parameters(parameters){};

    void setPositionInWindow(const moon::math::Vector<float,2>& offset, const moon::math::Vector<float,2>& size);
    void setParameters(const RayTracingLinkParameters& parameters){
        this->parameters = parameters;
    };

    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
    void setRenderPass(VkRenderPass renderPass) override;

    void create(const std::filesystem::path& shadersPath, VkDevice device, const moon::utils::ImageInfo& imageInfo);
    void updateDescriptorSets(const moon::utils::AttachmentsDatabase& aDatabase);
};

}
#endif
