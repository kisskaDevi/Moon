#ifndef LAYERSCOMBINER_H
#define LAYERSCOMBINER_H

#include "workflow.h"

class camera;

struct layersCombinerPushConst{
    alignas(4) int enableScatteringRefraction{true};
};

struct layersCombinerAttachments{
    attachments color;
    attachments bloom;

    inline const uint32_t size() const{
        return 2;
    }
    inline attachments* operator&(){
        return &color;
    }
    void deleteAttachment(VkDevice device){
        color.deleteAttachment(device);
        bloom.deleteAttachment(device);
    }
    void deleteSampler(VkDevice device){
        color.deleteSampler(device);
        bloom.deleteSampler(device);
    }
};

class layersCombiner : public workflow
{
private:
    layersCombinerAttachments frame;
    bool enable{true};

    struct Combiner : public workbody{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        uint32_t transparentLayersCount{1};
        bool enableScatteringRefraction{true};
    }combiner;

    void createAttachments(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    layersCombiner(bool enable, uint32_t transparentLayersCount, bool enableScatteringRefraction);

    void destroy() override;
    void create(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap) override;
    void updateDescriptorSets(
        const std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>>& bufferMap,
        const std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap) override;
    void updateCommandBuffer(uint32_t frameNumber) override;

    void setTransparentLayersCount(uint32_t transparentLayersCount);
    void setScatteringRefraction(bool enable);
};

#endif // LAYERSCOMBINER_H
