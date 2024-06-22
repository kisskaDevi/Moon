#ifndef SELECTOR_H
#define SELECTOR_H

#include "workflow.h"

namespace moon::workflows {

struct SelectorParameters{
    struct{
        std::string storageBuffer;
        std::string position;
        std::string depth;
        std::string transparency;
        std::string defaultDepthTexture;
    }in;
    struct{
        std::string selector;
    }out;
};

class SelectorGraphics : public Workflow
{
private:
    SelectorParameters parameters;

    moon::utils::Attachments frame;
    bool enable{true};

    struct Selector : public Workbody{
        uint32_t transparentLayersCount{ 1 };

        Selector(const moon::utils::ImageInfo& imageInfo) : Workbody(imageInfo) {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout();
    }selector;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
public:
    SelectorGraphics(const moon::utils::ImageInfo& imageInfo, const std::filesystem::path& shadersPath, SelectorParameters parameters, bool enable, uint32_t transparentLayersCount = 1);

    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // SELECTOR_H
