#ifndef SKYBOX_H
#define SKYBOX_H

#include "filtergraphics.h"

class object;
class camera;

class skyboxGraphics : public filterGraphics
{
private:
    struct Skybox : public filter{
        std::string                     vertShaderPath;
        std::string                     fragShaderPath;

        VkDescriptorSetLayout           ObjectDescriptorSetLayout{VK_NULL_HANDLE};

        std::vector<object*>            objects;

        void destroy(VkDevice device);
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }skybox;

public:
    skyboxGraphics() = default;
    void destroy();

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments);
    void createRenderPass() override;
    void createFramebuffers() override;
    void createPipelines() override;

    void createDescriptorPool() override;
    void createDescriptorSets() override;
    void updateDescriptorSets(camera* cameraObject);

    void updateCommandBuffer(uint32_t frameNumber) override;

    void bindObject(object* newObject);
    bool removeObject(object* object);

    void updateObjectUniformBuffer(VkCommandBuffer commandBuffer, uint32_t currentImage);
};

#endif // SKYBOX_H
