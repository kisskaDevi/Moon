#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include "workflow.h"
#include "vkdefault.h"

namespace moon::interfaces { class Object;}

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

    struct BoundingBox : public Workbody{
        moon::utils::vkDefault::DescriptorSetLayout objectDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout primitiveDescriptorSetLayout;

        std::vector<moon::interfaces::Object*>* objects;

        BoundingBox(const moon::utils::ImageInfo& imageInfo) : Workbody(imageInfo) {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout();
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }box;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
public:
    BoundingBoxGraphics(const moon::utils::ImageInfo& imageInfo, const std::filesystem::path& shadersPath, BoundingBoxParameters parameters, bool enable, std::vector<moon::interfaces::Object*>* objects = nullptr);

    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // BOUNDINGBOX_H
