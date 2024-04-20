#ifndef SCATTERING_H
#define SCATTERING_H

#include "workflow.h"

class light;
class depthMap;

struct scatteringPushConst{
    alignas(4) uint32_t  width{0};
    alignas(4) uint32_t  height{0};
};

class scattering : public workflow
{
private:
    attachments frame;
    bool enable{true};

    struct Lighting : workbody{
        VkDescriptorSetLayout                              ShadowDescriptorSetLayout{VK_NULL_HANDLE};
        std::unordered_map<uint8_t, VkDescriptorSetLayout> BufferDescriptorSetLayoutDictionary;
        std::unordered_map<uint8_t, VkDescriptorSetLayout> DescriptorSetLayoutDictionary;
        std::unordered_map<uint8_t, VkPipelineLayout>      PipelineLayoutDictionary;
        std::unordered_map<uint8_t, VkPipeline>            PipelinesDictionary;
        std::vector<light*>*                               lightSources;
        std::unordered_map<light*, depthMap*>*             depthMaps;

        void destroy(VkDevice device) override;
        void createPipeline(uint8_t mask, VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass);
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }lighting;

    void createAttachments(attachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    scattering(bool enable, std::vector<light*>* lightSources = nullptr,
               std::unordered_map<light*, depthMap*>* depthMaps = nullptr);

    void destroy() override;
    void create(attachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(
        const std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>>& bufferMap,
        const attachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

#endif // SCATTERING_H
