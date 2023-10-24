#ifndef SCATTERING_H
#define SCATTERING_H

#include "workflow.h"

class light;
class camera;

struct scatteringPushConst{
    alignas(4) int  width{0};
    alignas(4) int  height{0};
};

class scattering : public workflow
{
private:
    attachments frame;
    bool enable{true};

    struct Lighting : workbody{
        std::unordered_map<uint8_t, VkDescriptorSetLayout> BufferDescriptorSetLayoutDictionary;
        std::unordered_map<uint8_t, VkDescriptorSetLayout> DescriptorSetLayoutDictionary;
        std::unordered_map<uint8_t, VkPipelineLayout>      PipelineLayoutDictionary;
        std::unordered_map<uint8_t, VkPipeline>            PipelinesDictionary;
        std::vector<light*>     lightSources;

        void destroy(VkDevice device);
        void createPipeline(uint8_t mask, VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass);
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }lighting;

    void createAttachments(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    scattering(bool enable);

    void destroy() override;
    void create(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap) override;
    void updateDescriptorSets(
        const std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>>& bufferMap,
        const std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap) override;
    void updateCommandBuffer(uint32_t frameNumber) override;

    void bindLightSource(light* lightSource);
    bool removeLightSource(light* lightSource);
};

#endif // SCATTERING_H
