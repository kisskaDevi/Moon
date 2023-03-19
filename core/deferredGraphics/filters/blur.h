#ifndef BLUR_H
#define BLUR_H

#include "filtergraphics.h"

class gaussianBlur : public filterGraphics
{
private:
    attachments                         bufferAttachment;

    struct blur : public filter{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        uint32_t                        subpassNumber{0};
    };
    blur xblur;
    blur yblur;

public:
    gaussianBlur() = default;
    void destroy();

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments);
    void createRenderPass() override;
    void createFramebuffers() override;
    void createPipelines() override;

    void createDescriptorPool() override;
    void createDescriptorSets() override;
    void updateDescriptorSets(attachments* blurAttachment);

    void updateCommandBuffer(uint32_t frameNumber) override;

    void createBufferAttachments();
};

#endif // BLUR_H
