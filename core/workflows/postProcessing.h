#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include "workflow.h"

struct postProcessingParameters{
    struct{
        std::string baseColor;
        std::string blur;
        std::string bloom;
        std::string ssao;
        std::string boundingBox;
    }in;
    struct{
        std::string postProcessing;
    }out;
};

class postProcessingGraphics : public workflow
{
private:
    postProcessingParameters parameters;

    attachments frame;
    bool enable{true};

    struct PostProcessing : public workbody{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }postProcessing;

    void createAttachments(attachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    postProcessingGraphics(postProcessingParameters parameters, bool enable);

    void destroy() override;
    void create(attachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const buffersDatabase& bDatabase, const attachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

#endif // POSTPROCESSING_H
