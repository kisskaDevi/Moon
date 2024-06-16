#ifndef SSAO_H
#define SSAO_H

#include "workflow.h"

namespace moon::workflows {

struct SSAOParameters{
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
    SSAOParameters parameters;

    moon::utils::Attachments frame;
    bool enable{true};

    struct SSAO : public Workbody{
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device)override;
    }ssao;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    SSAOGraphics(SSAOParameters parameters, bool enable);
    ~SSAOGraphics() { destroy(); }

    void destroy();
    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // SSAO_H
