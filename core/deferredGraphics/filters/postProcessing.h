#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include "filtergraphics.h"

namespace SwapChain{
    struct SupportDetails;
}
class GLFWwindow;
class texture;

struct postProcessingPushConst{
    alignas(4) float                    blitFactor;
};

class postProcessingGraphics : public filterGraphics
{
private:
    uint32_t                            swapChainAttachmentCount{1};
    std::vector<attachments>            swapChainAttachments;

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
    void destroySwapChainAttachments();

    void createSwapChain(VkSwapchainKHR* swapChain, GLFWwindow* window, SwapChain::SupportDetails* swapChainSupport, VkSurfaceKHR* surface, uint32_t queueFamilyIndexCount, uint32_t* pQueueFamilyIndices);
    void createSwapChainAttachments(VkSwapchainKHR* swapChain);
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
