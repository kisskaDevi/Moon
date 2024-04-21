#ifndef BLUR_H
#define BLUR_H

#include "workflow.h"

struct gaussianBlurParameters{
    struct{
        std::string blur;
    }in;
    struct{
        std::string blur;
    }out;
};

class gaussianBlur : public workflow
{
private:
    gaussianBlurParameters parameters;

    attachments bufferAttachment;
    attachments frame;
    bool enable{true};

    float blurDepth{1.0f};

    struct blur : public workbody{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        uint32_t subpassNumber{0};
    };
    blur xblur;
    blur yblur;

    void createAttachments(attachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();

    void createBufferAttachments();
public:
    gaussianBlur(gaussianBlurParameters parameters, bool enable);

    void destroy() override;
    void create(attachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const buffersDatabase&, const attachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;

    gaussianBlur& setBlurDepth(float blurDepth);
};

#endif // BLUR_H
