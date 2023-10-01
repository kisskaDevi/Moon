#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include "workflow.h"

class object;
class camera;

class boundingBoxGraphics : public workflow
{
private:
    struct boundingBox : workbody{
        VkDescriptorSetLayout   ObjectDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout   PrimitiveDescriptorSetLayout{VK_NULL_HANDLE};

        std::vector<object*>    objects;

        void destroy(VkDevice device);
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }box;

public:
    boundingBoxGraphics() = default;
    void destroy();

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments);

    void createRenderPass()override;
    void createFramebuffers()override;
    void createPipelines()override;

    void createDescriptorPool()override;
    void createDescriptorSets()override;
    void updateDescriptorSets(camera* cameraObject);

    void updateCommandBuffer(uint32_t frameNumber) override;

    void bindObject(object* object);
    bool removeObject(object* object);
};

#endif // BOUNDINGBOX_H
