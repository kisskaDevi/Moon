#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include<libs/vulkan/vulkan.h>
#include "filtergraphics.h"
#include "core/operations.h"

#include <string>

struct SwapChainSupportDetails;
class GLFWwindow;
class texture;

struct postProcessingPushConst{
    alignas(4) float                    blitFactor;
};

class postProcessingGraphics : public filterGraphics
{
private:
    texture*                            emptyTexture{nullptr};

    uint32_t                            swapChainAttachmentCount{1};
    std::vector<attachments>            swapChainAttachments;

    attachments*                        blurAttachment{nullptr};
    attachments*                        blitAttachments{nullptr};
    attachments*                        sslrAttachment{nullptr};
    attachments*                        ssaoAttachment{nullptr};
    attachments*                        layersAttachment{nullptr};

    VkRenderPass                        renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>          framebuffers;

    struct PostProcessing : public filter{
        std::string                     vertShaderPath;
        std::string                     fragShaderPath;
        float                           blitFactor;
        uint32_t                        blitAttachmentCount;

        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }postProcessing;

public:
    postProcessingGraphics();
    void destroy()override;
    void destroySwapChainAttachments();

    void createSwapChain(VkSwapchainKHR* swapChain, GLFWwindow* window, SwapChain::SupportDetails swapChainSupport, VkSurfaceKHR* surface, uint32_t queueFamilyIndexCount, uint32_t* pQueueFamilyIndices);
    void createSwapChainAttachments(VkSwapchainKHR* swapChain);
    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments) override{
        static_cast<void>(attachmentsCount);
        static_cast<void>(pAttachments);
    };
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
