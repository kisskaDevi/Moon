#ifndef SCATTERING_H
#define SCATTERING_H

#include "workflow.h"
#include "vkdefault.h"

namespace moon::interfaces { class Light;}
namespace moon::utils { class DepthMap;}

namespace moon::workflows {

struct ScatteringParameters{
    struct{
        std::string camera;
        std::string depth;
    }in;
    struct{
        std::string scattering;
    }out;
};

class Scattering : public Workflow
{
private:
    ScatteringParameters parameters;

    moon::utils::Attachments frame;
    bool enable{true};

    struct Lighting : Workbody{
        moon::utils::vkDefault::DescriptorSetLayout     shadowDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayoutMap  bufferDescriptorSetLayoutMap;
        moon::utils::vkDefault::DescriptorSetLayoutMap  descriptorSetLayoutMap;

        moon::utils::vkDefault::PipelineLayoutMap       pipelineLayoutMap;
        moon::utils::vkDefault::PipelineMap             pipelinesMap;

        std::vector<moon::interfaces::Light*>* lightSources{ nullptr };
        std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>* depthMaps{ nullptr };

        Lighting(const moon::utils::ImageInfo& imageInfo) : Workbody(imageInfo) {};

        void createPipeline(uint8_t mask, VkRenderPass pRenderPass);
        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }lighting;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
public:
    Scattering(const moon::utils::ImageInfo& imageInfo, const std::filesystem::path& shadersPath, ScatteringParameters parameters,
               bool enable, std::vector<moon::interfaces::Light*>* lightSources = nullptr,
               std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>* depthMaps = nullptr);

    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(
        const moon::utils::BuffersDatabase& bDatabase,
        const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // SCATTERING_H
