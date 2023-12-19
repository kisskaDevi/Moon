#ifndef CUSTOMFILTER_H
#define CUSTOMFILTER_H

#include "workflow.h"

struct CustomFilterPushConst{
    alignas (4) float deltax;
    alignas (4) float deltay;
};

class customFilter : public workflow
{
private:
    std::vector<attachments> frames;
    attachments bufferAttachment;
    attachments* srcAttachment{nullptr};

    bool enable{true};
    float blitFactor{0.0f};
    float xSampleStep{1.5f};
    float ySampleStep{1.5f};
    uint32_t blitAttachmentsCount{0};

    struct Filter : public workbody{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }filter;

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, uint32_t attachmentNumber);

    void createAttachments(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    customFilter(bool enable, float blitFactor, float xSampleStep, float ySampleStep, uint32_t blitAttachmentsCount);

    void destroy() override;
    void create(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap) override;
    void updateDescriptorSets(
        const std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>>& bufferMap,
        const std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap) override;
    void updateCommandBuffer(uint32_t frameNumber) override;

    void setBlitFactor(const float& blitFactor);
    void setSampleStep(const float& deltaX, const float& deltaY);
};


#endif // CUSTOMFILTER_H
