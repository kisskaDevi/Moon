#ifndef LAYERSCOMBINER_H
#define LAYERSCOMBINER_H

#include "filtergraphics.h"

class camera;

class layersCombiner : public filterGraphics
{
private:
    VkRenderPass                        renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>          framebuffers;

    struct Combiner : public filter{
        std::string                     vertShaderPath;
        std::string                     fragShaderPath;
        uint32_t                        transparentLayersCount{0};

        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }combiner;

public:
    layersCombiner();
    void destroy() override;

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments) override;
    void createRenderPass() override;
    void createFramebuffers() override;
    void createPipelines() override;

    void createDescriptorPool() override;
    void createDescriptorSets() override;
    void updateDescriptorSets(DeferredAttachments deferredAttachments, DeferredAttachments* transparencyLayers, attachments* skybox, camera* cameraObject);

    void updateCommandBuffer(uint32_t frameNumber) override;

    void setTransparentLayersCount(uint32_t transparentLayersCount);
};

#endif // LAYERSCOMBINER_H
