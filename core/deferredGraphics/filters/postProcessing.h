#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include "filtergraphics.h"

namespace SwapChain{
    struct SupportDetails;
}
class GLFWwindow;
class texture;
class swapChain;

struct postProcessingPushConst{
    alignas(4) float                    blitFactor;
};

class postProcessingGraphics : public filterGraphics
{
private:
    swapChain*                          swapChainKHR{nullptr};
    attachments*                        blurAttachment{nullptr};
    attachments*                        blitAttachments{nullptr};
    attachments*                        sslrAttachment{nullptr};
    attachments*                        ssaoAttachment{nullptr};
    attachments*                        layersAttachment{nullptr};

    struct PostProcessing : public filter{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        float                           blitFactor;
        uint32_t                        blitAttachmentCount;
    }postProcessing;

public:
    postProcessingGraphics() = default;
    void destroy();

    void createRenderPass()override;
    void createFramebuffers()override;
    void createPipelines()override;

    void createDescriptorPool()override;
    void createDescriptorSets()override;
    void updateDescriptorSets();

    void updateCommandBuffer(uint32_t frameNumber) override;

    void setSwapChain(swapChain* swapChainKHR);
    void setBlurAttachment(attachments* blurAttachment);
    void setBlitAttachments(uint32_t blitAttachmentCount, attachments* blitAttachments, float blitFactor);
    void setSSLRAttachment(attachments* sslrAttachment);
    void setSSAOAttachment(attachments* ssaoAttachment);
    void setLayersAttachment(attachments* layersAttachment);
};

#endif // POSTPROCESSING_H
