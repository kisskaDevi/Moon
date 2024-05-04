#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include "workflow.h"

class object;

namespace moon::workflows {

struct BoundingBoxParameters{
    struct{
        std::string camera;
    }in;
    struct{
        std::string boundingBox;
    }out;
};

class BoundingBoxGraphics : public Workflow
{
private:
    BoundingBoxParameters parameters;

    moon::utils::Attachments frame;
    bool enable{true};

    struct BoundingBox : Workbody{
        VkDescriptorSetLayout   ObjectDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout   PrimitiveDescriptorSetLayout{VK_NULL_HANDLE};

        std::vector<object*>*   objects;

        void destroy(VkDevice device) override;
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }box;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    BoundingBoxGraphics(BoundingBoxParameters parameters, bool enable, std::vector<object*>* objects = nullptr);

    void destroy() override;
    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(
        const moon::utils::BuffersDatabase& bDatabase,
        const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // BOUNDINGBOX_H
