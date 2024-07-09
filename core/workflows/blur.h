#ifndef BLUR_H
#define BLUR_H

#include "workflow.h"

namespace moon::workflows {

struct GaussianBlurParameters : workflows::Parameters {
    struct{
        std::string blur;
    }in;
    struct{
        std::string blur;
    }out;
    float blurDepth{ 1.0f };
};

class GaussianBlur : public Workflow
{
private:
    GaussianBlurParameters& parameters;
    moon::utils::Attachments bufferAttachment;
    moon::utils::Attachments frame;

    struct Blur : public Workbody{
        uint32_t subpassNumber{ 0 };
        const GaussianBlurParameters& parameters;
        Blur(const moon::utils::ImageInfo& imageInfo, const GaussianBlurParameters& parameters, uint32_t subpassNumber)
            : Workbody(imageInfo), parameters(parameters), subpassNumber(subpassNumber)
        {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
    };
    Blur xblur;
    Blur yblur;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();

public:
    GaussianBlur(const moon::utils::ImageInfo& imageInfo, const std::filesystem::path& shadersPath, GaussianBlurParameters& parameters);

    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase&, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // BLUR_H
