#ifndef SHADOW_H
#define SHADOW_H

#include "workflow.h"
#include <unordered_map>

class object;
class light;

class shadowGraphics : public workflow
{
private:
    std::unordered_map<light*,std::vector<VkFramebuffer>>   framebuffers;

    struct Shadow : public workbody{
        void destroy(VkDevice device);
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        VkDescriptorSetLayout           lightUniformBufferSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout           ObjectDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout           PrimitiveDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout           MaterialDescriptorSetLayout{VK_NULL_HANDLE};

        std::vector<object*>            objects;
        std::vector<light*>             lightSources;
    }shadow;

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, uint32_t attachmentNumber);
public:
    shadowGraphics() = default;
    void destroy();

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments);
    void createRenderPass() override;
    void createFramebuffers() override {}
    void createPipelines() override;

    void createDescriptorPool() override {}
    void createDescriptorSets() override {}

    void updateCommandBuffer(uint32_t frameNumber) override;

    void createFramebuffers(light* lightSource);

    void addLightSource(light* lightSource);
    void removeLightSource(light* lightSource);

    void bindBaseObject(object* newObject);
    bool removeBaseObject(object* object);
};

#endif // SHADOW_H
