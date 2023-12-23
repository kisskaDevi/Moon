#ifndef SHADOW_H
#define SHADOW_H

#include "workflow.h"
#include <unordered_map>

class object;
class light;

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

        std::vector<object*>*   objects;
        std::vector<light*>*    lightSources;
    }shadow;

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, uint32_t attachmentNumber);
    void createRenderPass();
    void createPipelines();
public:
    shadowGraphics(bool enable, std::vector<object*>* objects = nullptr, std::vector<light*>* lightSources = nullptr);

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
