#ifndef SELECTOR_H
#define SELECTOR_H

#include "workflow.h"

struct selectorParameters{
    struct{
        std::string storageBuffer;
        std::string position;
        std::string depth;
        std::string transparency;
    }in;
    struct{
        std::string selector;
    }out;
};

class selectorGraphics : public workflow
{
private:
    selectorParameters parameters;

    attachments frame;
    bool enable{true};

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
    selectorGraphics(selectorParameters parameters, bool enable, uint32_t transparentLayersCount = 1);

    void destroy() override;
    void create(attachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const buffersDatabase& bDatabase, const attachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

#endif // SELECTOR_H
