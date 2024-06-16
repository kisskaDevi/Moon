#ifndef SKYBOX_H
#define SKYBOX_H

#include "workflow.h"
#include "vkdefault.h"

namespace moon::interfaces {class Object;}

namespace moon::workflows {

struct SkyboxAttachments
{
    moon::utils::Attachments color;
    moon::utils::Attachments bloom;

    inline uint32_t size() const{
        return 2;
    }
    inline moon::utils::Attachments* operator&(){
        return &color;
    }
    void deleteAttachment(VkDevice device){
        color.deleteAttachment(device);
        bloom.deleteAttachment(device);
    }
    void deleteSampler(VkDevice device){
        color.deleteSampler(device);
        bloom.deleteSampler(device);
    }
};

struct SkyboxParameters{
    struct{
        std::string camera;
    }in;
    struct{
        std::string baseColor;
        std::string bloom;
    }out;
};

class SkyboxGraphics : public Workflow
{
private:
    SkyboxParameters parameters;

    SkyboxAttachments frame;
    bool enable;

    struct Skybox : public Workbody{
        moon::utils::vkDefault::DescriptorSetLayout objectDescriptorSetLayout;

        std::vector<moon::interfaces::Object*>*   objects;

        void destroy(VkDevice device) override;
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }skybox;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    SkyboxGraphics(SkyboxParameters parameters, bool enable, std::vector<moon::interfaces::Object*>* object = nullptr);
    ~SkyboxGraphics() { destroy(); }

    void destroy();
    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // SKYBOX_H
