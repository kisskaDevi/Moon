#ifndef BLUR_H
#define BLUR_H

#include "workflow.h"

namespace moon::workflows {

struct GaussianBlurParameters{
    struct{
        std::string blur;
    }in;
    struct{
        std::string blur;
    }out;
};

class GaussianBlur : public Workflow
{
private:
    GaussianBlurParameters parameters;

    moon::utils::Attachments bufferAttachment;
    moon::utils::Attachments frame;
    bool enable{true};

    float blurDepth{1.0f};

    struct Blur : public Workbody{
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        uint32_t subpassNumber{0};
    };
    Blur xblur;
    Blur yblur;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();

public:
    GaussianBlur(GaussianBlurParameters parameters, bool enable);
    ~GaussianBlur() { destroy(); }

    void destroy();
    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase&, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;

    GaussianBlur& setBlurDepth(float blurDepth);
};

}
#endif // BLUR_H
