#ifndef SSLR_H
#define SSLR_H

#include "workflow.h"

namespace moon::workflows {

struct SSLRParameters : workflows::Parameters{
    struct{
        std::string camera;
        std::string position;
        std::string normal;
        std::string color;
        std::string depth;
        std::string firstTransparency;
        std::string defaultDepthTexture;
    }in;
    struct{
        std::string sslr;
    }out;
};

class SSLRGraphics : public Workflow
{
private:
    SSLRParameters& parameters;
    moon::utils::Attachments frame;

    struct SSLR : public Workbody{
        SSLR(const moon::utils::ImageInfo& imageInfo) : Workbody(imageInfo) {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
    }sslr;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
public:
    SSLRGraphics(const moon::utils::ImageInfo& imageInfo, const std::filesystem::path& shadersPath, SSLRParameters& parameters);

    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // SSLR_H
