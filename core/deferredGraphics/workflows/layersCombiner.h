#ifndef LAYERSCOMBINER_H
#define LAYERSCOMBINER_H

#include "workflow.h"
#include "deferredAttachments.h"

class camera;

struct layersCombinerPushConst{
    alignas(4) int enableScatteringRefraction{true};
};

class layersCombiner : public workflow
{
private:
    struct Combiner : public workbody{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        uint32_t transparentLayersCount{0};
        bool enableScatteringRefraction{true};
    }combiner;

public:
    layersCombiner() = default;
    void destroy();

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments);
    void createRenderPass() override;
    void createFramebuffers() override;
    void createPipelines() override;

    void createDescriptorPool() override;
    void createDescriptorSets() override;
    void updateDescriptorSets(DeferredAttachments deferredAttachments, DeferredAttachments* transparencyLayers, attachments* skybox, attachments* skyboxBloom, attachments* scattering, camera* cameraObject);

    void updateCommandBuffer(uint32_t frameNumber) override;

    void setTransparentLayersCount(uint32_t transparentLayersCount);
    void setScatteringRefraction(bool enable);
};

#endif // LAYERSCOMBINER_H
