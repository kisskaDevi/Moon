#ifndef SSLR_H
#define SSLR_H

#include "filtergraphics.h"

class camera;

class SSLRGraphics : public filterGraphics
{
private:
    VkRenderPass                        renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>          framebuffers;

    struct SSLR : public filter{
        std::string                     vertShaderPath;
        std::string                     fragShaderPath;

        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }sslr;

public:
    SSLRGraphics();
    void destroy() override;

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments) override;
    void createRenderPass() override;
    void createFramebuffers() override;
    void createPipelines() override;

    void createDescriptorPool() override;
    void createDescriptorSets() override;
    void updateDescriptorSets(camera* cameraObject, DeferredAttachments deferredAttachments, DeferredAttachments firstLayer);

    void updateCommandBuffer(uint32_t frameNumber) override;
};

#endif // SSLR_H
