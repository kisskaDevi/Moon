#ifndef SELECTOR_H
#define SELECTOR_H

#include "workflow.h"
#include "cursor.h"

namespace moon::workflows {

struct SelectorParameters : workflows::Parameters {
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
    uint32_t transparentLayersCount{ 1 };
};

class SelectorGraphics : public Workflow
{
private:
    SelectorParameters& parameters;
    moon::utils::Attachments frame;
    utils::Cursor** cursor{ nullptr };

    struct Selector : public Workbody{
        const SelectorParameters& parameters;
        Selector(const moon::utils::ImageInfo& imageInfo, const SelectorParameters& parameters)
            : Workbody(imageInfo), parameters(parameters)
        {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
    }selector;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    SelectorGraphics(SelectorParameters& parameters, utils::Cursor** cursor);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

}
#endif // SELECTOR_H
