#ifndef BLUR_H
#define BLUR_H

#include "filtergraphics.h"

class gaussianBlur : public filterGraphics
{
private:
    attachments                         bufferAttachment;

    VkRenderPass                        renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>          framebuffers;

    struct blur : public filter{
        std::string                     vertShaderPath;
        std::string                     fragShaderPath;
        uint32_t                        subpassNumber;

        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    };
    blur xblur;
    blur yblur;

public:
    gaussianBlur();
    void destroy() override;

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments) override;
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
