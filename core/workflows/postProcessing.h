#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include "workflow.h"

namespace moon::workflows {

struct PostProcessingParameters : workflows::Parameters{
    struct{
        std::string baseColor;
        std::string blur;
        std::string bloom;
        std::string ssao;
        std::string boundingBox;
    }in;
    struct{
        std::string postProcessing;
    }out;
};

class PostProcessingGraphics : public Workflow
{
private:
    PostProcessingParameters& parameters;
    moon::utils::Attachments frame;

    struct PostProcessing : public Workbody{
        PostProcessing(const moon::utils::ImageInfo& imageInfo) : Workbody(imageInfo) {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
    } postProcessing;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    PostProcessingGraphics(PostProcessingParameters& parameters);

    void create(const utils::vkDefault::CommandPool& commandPool, moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
};

}
#endif // POSTPROCESSING_H
