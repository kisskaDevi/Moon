#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include "workflow.h"

namespace SwapChain{
    struct SupportDetails;
}
struct GLFWwindow;
class texture;
class swapChain;

struct postProcessingPushConst{
    alignas(4) float blitFactor;
};

class postProcessingGraphics : public workflow
{
private:
    attachments*    blurAttachment{nullptr};
    attachments*    blitAttachments{nullptr};
    attachments*    sslrAttachment{nullptr};
    attachments*    ssaoAttachment{nullptr};
    attachments*    layersAttachment{nullptr};

    struct PostProcessing : public workbody{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        float       blitFactor;
        uint32_t    blitAttachmentCount;
    }postProcessing;

public:
    postProcessingGraphics() = default;
    void destroy();

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments);

    void createRenderPass()override;
    void createFramebuffers()override;
    void createPipelines()override;

    void createDescriptorPool()override;
    void createDescriptorSets()override;
    void updateDescriptorSets();

    void updateCommandBuffer(uint32_t frameNumber) override;

    void setBlurAttachment(attachments* blurAttachment);
    void setBlitAttachments(uint32_t blitAttachmentCount, attachments* blitAttachments, float blitFactor);
    void setSSLRAttachment(attachments* sslrAttachment);
    void setSSAOAttachment(attachments* ssaoAttachment);
    void setLayersAttachment(attachments* layersAttachment);
};

#endif // POSTPROCESSING_H
