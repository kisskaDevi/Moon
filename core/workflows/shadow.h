#ifndef SHADOW_H
#define SHADOW_H

#include "workflow.h"
#include "vkdefault.h"
#include <unordered_map>

namespace moon::interfaces {
class Object;
class Light;
}
namespace moon::utils { class DepthMap;}

namespace moon::workflows {

class ShadowGraphics : public Workflow
{
private:
    std::unordered_map<moon::utils::DepthMap*, utils::vkDefault::Framebuffers> framebuffersMap;
    bool enable{true};

    struct Shadow : public Workbody{
        void destroy(VkDevice device) override;
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        moon::utils::vkDefault::DescriptorSetLayout lightUniformBufferSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout objectDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout primitiveDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout materialDescriptorSetLayout;

        std::vector<moon::interfaces::Object*>* objects;
        std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>* depthMaps;
    }shadow;

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, moon::interfaces::Light* lightSource, moon::utils::DepthMap* depthMap);
    void createRenderPass();
    void createPipelines();
    moon::utils::Attachments* createAttachments();
public:
    ShadowGraphics(bool enable,
                   std::vector<moon::interfaces::Object*>* objects = nullptr,
                   std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>* depthMaps = nullptr);
    ~ShadowGraphics() { destroy(); }

    void destroy();
    void create(moon::utils::AttachmentsDatabase&) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase&, const moon::utils::AttachmentsDatabase&) override{};
    void updateCommandBuffer(uint32_t frameNumber) override;

    void createFramebuffers(moon::utils::DepthMap* depthMap);
    void destroyFramebuffers(moon::utils::DepthMap* depthMap);
};

}
#endif // SHADOW_H
