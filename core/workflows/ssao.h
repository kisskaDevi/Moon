#ifndef SSAO_H
#define SSAO_H

#include "workflow.h"

namespace moon::workflows {

struct SSAOParameters : workflows::Parameters{
    struct{
        std::string camera;
        std::string position;
        std::string normal;
        std::string color;
        std::string depth;
        std::string defaultDepthTexture;
    }in;
    struct{
        std::string ssao;
    }out;
};

class SSAOGraphics : public Workflow
{
private:
    SSAOParameters& parameters;
    moon::utils::Attachments frame;

    struct SSAO : public Workbody{
        SSAO(const moon::utils::ImageInfo& imageInfo) : Workbody(imageInfo) {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
    }ssao;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
public:
    SSAOGraphics(SSAOParameters& parameters);

    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // SSAO_H
