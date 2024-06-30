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
    uint32_t                        imageCount{0};
    VkDevice                        device{VK_NULL_HANDLE};
    VkRenderPass                    renderPass{VK_NULL_HANDLE};

    VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
    VkPipeline                      Pipeline{VK_NULL_HANDLE};
    VkDescriptorSetLayout           DescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
    utils::vkDefault::DescriptorSets DescriptorSets;

    LinkPushConstant                pushConstant;

public:
    RayTracingLink() = default;
    RayTracingLink(const RayTracingLinkParameters& parameters) : parameters(parameters){};
    void destroy();

    void setDeviceProp(VkDevice device);
    void setImageCount(const uint32_t& count);
    void setShadersPath(const std::filesystem::path& shadersPath);
    void setPositionInWindow(const moon::math::Vector<float,2>& offset, const moon::math::Vector<float,2>& size);
    void setParameters(const RayTracingLinkParameters& parameters){
        this->parameters = parameters;
    };

    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
    void setRenderPass(VkRenderPass renderPass) override;

    void createDescriptorSetLayout();
    void createPipeline(moon::utils::ImageInfo* pInfo);
    void createDescriptorPool();
    void createDescriptorSets();
    void updateDescriptorSets(const moon::utils::AttachmentsDatabase& aDatabase);
};

}
#endif
