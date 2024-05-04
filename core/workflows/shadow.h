#ifndef SHADOW_H
#define SHADOW_H

#include "workflow.h"
#include <unordered_map>

class object;
class light;
namespace moon::utils { class DepthMap;}

class shadowGraphics : public workflow
{
private:
    std::unordered_map<moon::utils::DepthMap*,std::vector<VkFramebuffer>> framebuffers;
    bool enable{true};

    struct Shadow : public workbody{
        void destroy(VkDevice device) override;
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        VkDescriptorSetLayout   lightUniformBufferSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout   ObjectDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout   PrimitiveDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout   MaterialDescriptorSetLayout{VK_NULL_HANDLE};

        std::vector<object*>* objects;
        std::unordered_map<light*, moon::utils::DepthMap*>* depthMaps;
    }shadow;

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, light* lightSource, moon::utils::DepthMap* depthMap);
    void createRenderPass();
    void createPipelines();
    moon::utils::Attachments* createAttachments();
public:
    shadowGraphics(bool enable,
                   std::vector<object*>* objects = nullptr,
                   std::unordered_map<light*, moon::utils::DepthMap*>* depthMaps = nullptr);

    void destroy() override;
    void create(moon::utils::AttachmentsDatabase&) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase&, const moon::utils::AttachmentsDatabase&) override{};
    void updateCommandBuffer(uint32_t frameNumber) override;

    void createFramebuffers(moon::utils::DepthMap* depthMap);
    void destroyFramebuffers(moon::utils::DepthMap* depthMap);
};

#endif // SHADOW_H
