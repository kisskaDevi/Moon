#ifndef BLOOM_H
#define BLOOM_H

#include "workflow.h"

struct bloomPushConst{
    alignas (4) float deltax;
    alignas (4) float deltay;
    alignas (4) float blitFactor;
};

struct bloomParameters{
    struct{
        std::string bloom;
    }in;
    struct{
        std::string bloom;
    }out;
};

class bloomGraphics : public workflow
{
private:
    bloomParameters parameters;

    std::vector<moon::utils::Attachments> frames;
    moon::utils::Attachments bufferAttachment;
    const moon::utils::Attachments* srcAttachment{nullptr};

    bool enable{true};
    float blitFactor{1.5f};
    float xSamplerStep{1.5f};
    float ySamplerStep{1.5f};
    VkImageLayout inputImageLayout{VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    struct Filter : public workbody{
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }filter;

    struct Bloom : public workbody{
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        uint32_t blitAttachmentsCount{0};
    }bloom;

    void render(VkCommandBuffer commandBuffer, moon::utils::Attachments image, uint32_t frameNumber, uint32_t framebufferIndex, workbody* worker);

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    bloomGraphics() = default;
    bloomGraphics(bloomParameters parameters, bool enable, uint32_t blitAttachmentsCount, VkImageLayout inputImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, float blitFactor = 1.5f, float xSamplerStep = 1.5f, float ySamplerStep = 1.5f);

    void destroy() override;
    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase&, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;

    bloomGraphics& setBlitFactor(const float& blitFactor);
    bloomGraphics& setSamplerStepX(const float& xSamplerStep);
    bloomGraphics& setSamplerStepY(const float& ySamplerStep);
};


#endif // BLOOM_H
