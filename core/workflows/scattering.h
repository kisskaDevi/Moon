#ifndef SCATTERING_H
#define SCATTERING_H

#include "workflow.h"
#include "vkdefault.h"
#include "light.h"

namespace moon::workflows {

struct ScatteringParameters : workflows::Parameters{
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
    ScatteringParameters& parameters;
    moon::utils::Attachments frame;

    struct Lighting : Workbody{
        moon::utils::vkDefault::DescriptorSetLayout     shadowDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayoutMap  bufferDescriptorSetLayoutMap;
        moon::utils::vkDefault::DescriptorSetLayoutMap  descriptorSetLayoutMap;

        moon::utils::vkDefault::PipelineLayoutMap       pipelineLayoutMap;
        moon::utils::vkDefault::PipelineMap             pipelinesMap;

        const interfaces::Lights* lightSources{ nullptr };
        const interfaces::DepthMaps* depthMaps{ nullptr };

        Lighting(const moon::utils::ImageInfo& imageInfo, const interfaces::Lights* lightSources, const interfaces::DepthMaps* depthMaps)
            : Workbody(imageInfo), lightSources(lightSources), depthMaps(depthMaps)
        {}

        void createPipeline(uint8_t mask, VkRenderPass pRenderPass);
        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }lighting;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    Scattering(ScatteringParameters& parameters, const interfaces::Lights* lightSources = nullptr, const interfaces::DepthMaps* depthMaps = nullptr);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

}
#endif // SCATTERING_H
