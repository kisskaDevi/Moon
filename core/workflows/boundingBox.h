#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include "workflow.h"
#include "vkdefault.h"
#include "object.h"

namespace moon::workflows {

struct BoundingBoxParameters : workflows::Parameters {
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
    BoundingBoxParameters& parameters;
    moon::utils::Attachments frame;

    struct BoundingBox : public Workbody{
        moon::utils::vkDefault::DescriptorSetLayout objectDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout primitiveDescriptorSetLayout;

        const interfaces::Objects* objects;

        BoundingBox(const moon::utils::ImageInfo& imageInfo, const interfaces::Objects* objects)
            : Workbody(imageInfo), objects(objects)
        {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }box;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    BoundingBoxGraphics(BoundingBoxParameters& parameters, const interfaces::Objects* objects = nullptr);

    void create(const utils::vkDefault::CommandPool& commandPool, moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
};

}
#endif // BOUNDINGBOX_H
