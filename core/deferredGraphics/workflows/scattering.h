#ifndef SCATTERING_H
#define SCATTERING_H

#include "workflow.h"

class light;
class camera;

struct scatteringPushConst{
    alignas(4) int width{0};
    alignas(4) int height{0};
};

class scattering : public workflow
{
private:
    struct Lighting : workbody{
        VkDescriptorSetLayout  BufferDescriptorSetLayoutDictionary;
        VkDescriptorSetLayout  DescriptorSetLayoutDictionary;

        std::vector<light*> lightSources;

        void destroy(VkDevice device);
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }lighting;

public:
    scattering() = default;
    void destroy();

    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments);

    void createRenderPass()override;
    void createFramebuffers()override;
    void createPipelines()override;

    void createDescriptorPool()override;
    void createDescriptorSets()override;
    void updateDescriptorSets(camera* cameraObject, attachments* depth);

    void updateCommandBuffer(uint32_t frameNumber) override;

    void bindLightSource(light* lightSource);
    bool removeLightSource(light* lightSource);
};

#endif // SCATTERING_H
