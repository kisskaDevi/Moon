#ifndef SCATTERING_H
#define SCATTERING_H

#include "workflow.h"
#include "vkdefault.h"

namespace moon::interfaces { class Light;}
namespace moon::utils { class DepthMap;}

namespace moon::workflows {

struct ScatteringPushConst{
    alignas(4) uint32_t  width{0};
    alignas(4) uint32_t  height{0};
};

struct ScatteringParameters{
    struct{
        std::string camera;
        std::string depth;
    }in;
    struct{
        std::string scattering;
    }out;
};

class Scattering : public Workflow
{
private:
    ScatteringParameters parameters;

    moon::utils::Attachments frame;
    bool enable{true};

    struct Lighting : Workbody{
        moon::utils::vkDefault::DescriptorSetLayout     shadowDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayoutMap  bufferDescriptorSetLayoutMap;
        moon::utils::vkDefault::DescriptorSetLayoutMap  descriptorSetLayoutMap;

        moon::utils::vkDefault::PipelineLayoutMap       pipelineLayoutMap;
        moon::utils::vkDefault::PipelineMap             pipelinesMap;

        std::vector<moon::interfaces::Light*>*                                  lightSources;
        std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>*   depthMaps;

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
    Scattering(ScatteringParameters parameters,
               bool enable, std::vector<moon::interfaces::Light*>* lightSources = nullptr,
               std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>* depthMaps = nullptr);

    void destroy() override;
    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(
        const moon::utils::BuffersDatabase& bDatabase,
        const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // SCATTERING_H
