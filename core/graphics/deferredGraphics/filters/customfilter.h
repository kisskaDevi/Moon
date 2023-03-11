#ifndef CUSTOMFILTER_H
#define CUSTOMFILTER_H

#include "filtergraphics.h"

struct CustomFilterPushConst{
    alignas (4) float               deltax;
    alignas (4) float               deltay;
};

class customFilter : public filterGraphics
{
private:
    attachments                         bufferAttachment;
    attachments*                        srcAttachment{nullptr};
    float                               blitFactor{0.0f};
    float                               xSampleStep{1.5f};
    float                               ySampleStep{1.5f};

    VkRenderPass                        renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>          framebuffers;

    struct Filter : public filter{
        std::string                     vertShaderPath;
        std::string                     fragShaderPath;

        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }filter;

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, uint32_t attachmentNumber);
public:
    customFilter();
    void destroy() override;

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments) override;
    void createRenderPass() override;
    void createFramebuffers() override;
    void createPipelines() override;

    void createDescriptorPool() override;
    void createDescriptorSets() override;
    void updateDescriptorSets();

    void updateCommandBuffer(uint32_t frameNumber) override;

    void createBufferAttachments();
    void setSrcAttachment(attachments* srcAttachment);
    void setBlitFactor(const float& blitFactor);
    void setSampleStep(const float& deltaX, const float& deltaY);
};


#endif // CUSTOMFILTER_H
