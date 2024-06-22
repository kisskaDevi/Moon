#ifndef BLOOM_H
#define BLOOM_H

#include "workflow.h"

namespace moon::workflows {

struct BloomParameters{
    struct{
        std::string bloom;
    }in;
    struct{
        std::string bloom;
    }out;
};

class BloomGraphics : public Workflow
{
private:
    BloomParameters parameters;

    std::vector<moon::utils::Attachments> frames;
    moon::utils::Attachments bufferAttachment;
    const moon::utils::Attachments* srcAttachment{nullptr};

    bool enable{true};
    float blitFactor{1.5f};
    float xSamplerStep{1.5f};
    float ySamplerStep{1.5f};
    VkImageLayout inputImageLayout{VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    struct Filter : public Workbody{
        Filter() = default;
        Filter(Filter&&) = default;
        Filter& operator=(Filter&&) = default;

        Filter(const moon::utils::ImageInfo& imageInfo) : Workbody(imageInfo) {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout();
    }filter;

    struct Bloom : public Workbody{
        uint32_t blitAttachmentsCount{ 0 };

        Bloom() = default;
        Bloom(Bloom&&) = default;
        Bloom& operator=(Bloom&&) = default;

        Bloom(const moon::utils::ImageInfo& imageInfo) : Workbody(imageInfo) {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout();
    }bloom;

    void render(VkCommandBuffer commandBuffer, const moon::utils::Attachments& image, uint32_t frameNumber, uint32_t framebufferIndex, Workbody* worker);

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
public:
    BloomGraphics(BloomGraphics&&) = default;
    BloomGraphics& operator=(BloomGraphics&&) = default;
    BloomGraphics(const moon::utils::ImageInfo& imageInfo, const std::filesystem::path& shadersPath, BloomParameters parameters, bool enable, uint32_t blitAttachmentsCount, VkImageLayout inputImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, float blitFactor = 1.5f, float xSamplerStep = 1.5f, float ySamplerStep = 1.5f);
    BloomGraphics();

    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase&, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;

    BloomGraphics& setBlitFactor(const float& blitFactor);
    BloomGraphics& setSamplerStepX(const float& xSamplerStep);
    BloomGraphics& setSamplerStepY(const float& ySamplerStep);
};

}
#endif // BLOOM_H
