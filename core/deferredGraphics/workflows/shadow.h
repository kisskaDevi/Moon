#ifndef SHADOW_H
#define SHADOW_H

#include "workflow.h"
#include <unordered_map>

class object;
class light;
class depthMap;

class shadowGraphics : public workflow
{
private:
    std::unordered_map<light*,std::vector<VkFramebuffer>> framebuffers;
    bool enable{true};

    struct Shadow : public workbody{
        void destroy(VkDevice device);
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        VkDescriptorSetLayout   lightUniformBufferSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout   ObjectDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout   PrimitiveDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout   MaterialDescriptorSetLayout{VK_NULL_HANDLE};

        std::vector<object*>* objects;
        std::unordered_map<light*, depthMap*>* depthMaps;
    }shadow;

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, light* lightSource, depthMap* depthMap);
    void createRenderPass();
    void createPipelines();
public:
    shadowGraphics(bool enable,
                   std::vector<object*>* objects = nullptr,
                   std::unordered_map<light*, depthMap*>* depthMaps = nullptr);

    void destroy() override;
    void create(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap) override;
    void updateDescriptorSets(
        const std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>>&,
        const std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>&) override{};
    void updateCommandBuffer(uint32_t frameNumber) override;

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments);
    void createFramebuffers(light* lightSource);
    void destroyFramebuffers(light* lightSource);
};

#endif // SHADOW_H
