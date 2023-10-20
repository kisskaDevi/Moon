#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include "workflow.h"

namespace SwapChain{
    struct SupportDetails;
}
struct GLFWwindow;
class texture;
class swapChain;

struct postProcessingPushConst{
    alignas(4) float blitFactor;
};

class postProcessingGraphics : public workflow
{
private:
    attachments frame;
    bool enable{true};

    struct PostProcessing : public workbody{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        float       blitFactor{0.0f};
        uint32_t    blitAttachmentCount{1};
    }postProcessing;

    void createAttachments(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    postProcessingGraphics(bool enable, float blitFactor, uint32_t blitAttachmentCount);

    void destroy() override;
    void create(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap) override;
    void updateDescriptorSets(
        const std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>>& bufferMap,
        const std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

#endif // POSTPROCESSING_H
