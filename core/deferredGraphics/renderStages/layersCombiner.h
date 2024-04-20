#ifndef LAYERSCOMBINER_H
#define LAYERSCOMBINER_H

#include "workflow.h"

struct layersCombinerPushConst{
    alignas(4) int enableScatteringRefraction{true};
    alignas(4) int enableTransparentLayers{true};
    alignas(4) float blurDepth{1.0f};
};

struct layersCombinerAttachments{
    attachments color;
    attachments bloom;
    attachments blur;

    static inline uint32_t size() {
        return 3;
    }
    inline attachments* operator&(){
        return &color;
    }
    void deleteAttachment(VkDevice device){
        color.deleteAttachment(device);
        bloom.deleteAttachment(device);
        blur.deleteAttachment(device);
    }
    void deleteSampler(VkDevice device){
        color.deleteSampler(device);
        bloom.deleteSampler(device);
        blur.deleteSampler(device);
    }
};

class layersCombiner : public workflow
{
private:
    layersCombinerAttachments frame;
    bool enable{true};

    float blurDepth{1.0f};

    struct Combiner : public workbody{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        uint32_t transparentLayersCount{1};
        bool enableTransparentLayers{true};
        bool enableScatteringRefraction{true};
    }combiner;

    void createAttachments(attachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    layersCombiner(bool enable, uint32_t transparentLayersCount, bool enableScatteringRefraction);

    void destroy() override;
    void create(attachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(
        const std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>>& bufferMap,
        const attachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;

    void setTransparentLayersCount(uint32_t transparentLayersCount);
    void setScatteringRefraction(bool enable);
    void setBlurDepth(float blurDepth);
};

#endif // LAYERSCOMBINER_H
