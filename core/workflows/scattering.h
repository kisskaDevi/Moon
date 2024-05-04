#ifndef SCATTERING_H
#define SCATTERING_H

#include "workflow.h"

class light;
namespace moon::utils { class DepthMap;}

struct scatteringPushConst{
    alignas(4) uint32_t  width{0};
    alignas(4) uint32_t  height{0};
};

struct scatteringParameters{
    struct{
        std::string camera;
        std::string depth;
    }in;
    struct{
        std::string scattering;
    }out;
};

class scattering : public workflow
{
private:
    scatteringParameters parameters;

    moon::utils::Attachments frame;
    bool enable{true};

    struct Lighting : workbody{
        VkDescriptorSetLayout                               ShadowDescriptorSetLayout{VK_NULL_HANDLE};
        std::unordered_map<uint8_t, VkDescriptorSetLayout>  BufferDescriptorSetLayoutDictionary;
        std::unordered_map<uint8_t, VkDescriptorSetLayout>  DescriptorSetLayoutDictionary;
        std::unordered_map<uint8_t, VkPipelineLayout>       PipelineLayoutDictionary;
        std::unordered_map<uint8_t, VkPipeline>             PipelinesDictionary;
        std::vector<light*>*                                lightSources;
        std::unordered_map<light*, moon::utils::DepthMap*>* depthMaps;

        void destroy(VkDevice device) override;
        void createPipeline(uint8_t mask, VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass);
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }lighting;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    scattering(scatteringParameters parameters,
               bool enable, std::vector<light*>* lightSources = nullptr,
               std::unordered_map<light*, moon::utils::DepthMap*>* depthMaps = nullptr);

    void destroy() override;
    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(
        const moon::utils::BuffersDatabase& bDatabase,
        const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

#endif // SCATTERING_H
