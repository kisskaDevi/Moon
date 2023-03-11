#ifndef SKYBOX_H
#define SKYBOX_H

#include "filtergraphics.h"

class skyboxObject;
class camera;

class skyboxGraphics : public filterGraphics
{
private:
    VkRenderPass                        renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>          framebuffers;

    struct Skybox : public filter{
        std::string                     vertShaderPath;
        std::string                     fragShaderPath;

        VkDescriptorSetLayout           ObjectDescriptorSetLayout{VK_NULL_HANDLE};

        std::vector<skyboxObject*>      objects;

        void destroy(VkDevice device);
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }skybox;

public:
    skyboxGraphics();
    void destroy() override;

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments) override;
    void createRenderPass() override;
    void createFramebuffers() override;
    void createPipelines() override;

    void createDescriptorPool() override;
    void createDescriptorSets() override;
    void updateDescriptorSets(camera* cameraObject);

    void updateCommandBuffer(uint32_t frameNumber) override;

    void bindObject(skyboxObject* newObject);
    bool removeObject(skyboxObject* object);

    void updateObjectUniformBuffer(VkCommandBuffer commandBuffer, uint32_t currentImage);
};

#endif // SKYBOX_H
