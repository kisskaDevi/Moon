#ifndef SSLR_H
#define SSLR_H

#include "workflow.h"

struct SSLRParameters{
    struct{
        std::string camera;
        std::string position;
        std::string normal;
        std::string color;
        std::string depth;
        std::string firstTransparency;
    }in;
    struct{
        std::string sslr;
    }out;
};

class SSLRGraphics : public workflow
{
private:
    SSLRParameters parameters;

    attachments frame;
    bool enable;

    struct SSLR : public workbody{
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }sslr;

    void createAttachments(attachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    SSLRGraphics(SSLRParameters parameters, bool enable);

    void destroy() override;
    void create(attachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const buffersDatabase& bDatabase, const attachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

#endif // SSLR_H
