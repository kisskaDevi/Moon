#ifndef SELECTOR_H
#define SELECTOR_H

#include "workflow.h"

class selectorGraphics : public workflow
{
    attachments frame;
    bool enable{true};

private:
    struct Selector : public workbody{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device)override;

        uint32_t transparentLayersCount{1};
    }selector;

    void createAttachments(attachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    selectorGraphics(bool enable, uint32_t transparentLayersCount = 1);

    void destroy() override;
    void create(attachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(
        const std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>>& bufferMap,
        const attachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

#endif // SELECTOR_H
