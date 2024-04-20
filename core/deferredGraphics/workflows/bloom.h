#ifndef BLOOM_H
#define BLOOM_H

#include "workflow.h"

struct bloomPushConst{
    alignas (4) float deltax;
    alignas (4) float deltay;
    alignas (4) float blitFactor;
};

class bloomGraphics : public workflow
{
private:
    std::vector<attachments> frames;
    attachments bufferAttachment;
    const attachments* srcAttachment{nullptr};

    bool enable{true};
    float blitFactor{1.5f};
    float xSamplerStep{1.5f};
    float ySamplerStep{1.5f};

    struct Filter : public workbody{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }filter;

    struct Bloom : public workbody{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        uint32_t blitAttachmentsCount{0};
    }bloom;

    void render(VkCommandBuffer commandBuffer, attachments image, uint32_t frameNumber, uint32_t framebufferIndex, workbody* worker);

    void createAttachments(attachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    bloomGraphics(bool enable, uint32_t blitAttachmentsCount, float blitFactor = 1.5f, float xSamplerStep = 1.5f, float ySamplerStep = 1.5f);

    void destroy() override;
    void create(attachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const buffersDatabase&, const attachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;

    bloomGraphics& setBlitFactor(const float& blitFactor);
    bloomGraphics& setSamplerStepX(const float& xSamplerStep);
    bloomGraphics& setSamplerStepY(const float& ySamplerStep);
};


#endif // BLOOM_H
