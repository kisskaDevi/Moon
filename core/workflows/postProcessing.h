#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include "workflow.h"

namespace moon::workflows {

struct PostProcessingParameters{
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
    PostProcessingParameters parameters;

    moon::utils::Attachments frame;
    bool enable{true};

    struct PostProcessing : public Workbody{
        PostProcessing(const moon::utils::ImageInfo& imageInfo) : Workbody(imageInfo) {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout();
    } postProcessing;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
public:
    PostProcessingGraphics(const moon::utils::ImageInfo& imageInfo, const std::filesystem::path& shadersPath, PostProcessingParameters parameters, bool enable);

    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // POSTPROCESSING_H
