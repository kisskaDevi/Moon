#ifndef SSLR_H
#define SSLR_H

#include "workflow.h"

namespace moon::workflows {

struct SSLRParameters{
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
    SSLRParameters parameters;

    moon::utils::Attachments frame;
    bool enable;

    struct SSLR : public Workbody{
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }sslr;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    SSLRGraphics(SSLRParameters parameters, bool enable);
    ~SSLRGraphics() { destroy(); }

    void destroy();
    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // SSLR_H
