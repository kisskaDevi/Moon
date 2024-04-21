#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include "workflow.h"

class object;

struct boundingBoxParameters{
    struct{
        std::string camera;
    }in;
    struct{
        std::string boundingBox;
    }out;
};

class boundingBoxGraphics : public workflow
{
private:
    boundingBoxParameters parameters;

    attachments frame;
    bool enable{true};

    struct boundingBox : workbody{
        VkDescriptorSetLayout   ObjectDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout   PrimitiveDescriptorSetLayout{VK_NULL_HANDLE};

        std::vector<object*>*   objects;

        void destroy(VkDevice device) override;
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }box;

    void createAttachments(attachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    boundingBoxGraphics(boundingBoxParameters parameters, bool enable, std::vector<object*>* objects = nullptr);

    void destroy() override;
    void create(attachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(
        const buffersDatabase& bDatabase,
        const attachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

#endif // BOUNDINGBOX_H
