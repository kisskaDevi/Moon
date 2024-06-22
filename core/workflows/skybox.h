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
        std::vector<moon::interfaces::Object*>* objects{nullptr};

        Skybox(const moon::utils::ImageInfo& imageInfo) : Workbody(imageInfo) {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout();
    }skybox;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
public:
    SkyboxGraphics(const moon::utils::ImageInfo& imageInfo, const std::filesystem::path& shadersPath, SkyboxParameters parameters, bool enable, std::vector<moon::interfaces::Object*>* object = nullptr);

    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // SKYBOX_H
