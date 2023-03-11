#ifndef SHADOW_H
#define SHADOW_H

#include "filtergraphics.h"
#include <unordered_map>

class object;
class Node;
class light;

class shadowGraphics : public filterGraphics
{
private:
    std::unordered_map<light*,std::vector<VkFramebuffer>>   framebuffers;

    struct Shadow : public filter{
        void destroy(VkDevice device);
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;

        VkDescriptorSetLayout           lightUniformBufferSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout           uniformBufferSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout           uniformBlockSetLayout{VK_NULL_HANDLE};

        std::vector<object*>            objects;
        std::vector<light*>             lightSources;
    }shadow;

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, uint32_t attachmentNumber);
    void renderNode(VkCommandBuffer commandBuffer, Node* node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets);
public:
    shadowGraphics();
    void destroy();

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments) override;
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
