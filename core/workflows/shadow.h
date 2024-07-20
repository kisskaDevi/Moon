#ifndef SHADOW_H
#define SHADOW_H

#include "workflow.h"
#include "vkdefault.h"
#include "object.h"
#include "light.h"

namespace moon::workflows {

struct ShadowGraphicsParameters : workflows::Parameters {};

class ShadowGraphics : public Workflow
{
private:
    ShadowGraphicsParameters& parameters;
    std::unordered_map<const moon::utils::DepthMap*, moon::utils::vkDefault::Framebuffers> framebuffersMap;

    struct Shadow : public Workbody{
        Shadow(const moon::utils::ImageInfo& imageInfo, const interfaces::Objects* objects, interfaces::DepthMaps* depthMaps)
            : Workbody(imageInfo), objects(objects), depthMaps(depthMaps)
        {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;

        moon::utils::vkDefault::DescriptorSetLayout lightUniformBufferSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout objectDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout primitiveDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout materialDescriptorSetLayout;

        const interfaces::Objects* objects{ nullptr };
        interfaces::DepthMaps* depthMaps{ nullptr };
    }shadow;

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, interfaces::Light* lightSource, const utils::DepthMap& depthMap);
    void createRenderPass();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    ShadowGraphics(ShadowGraphicsParameters& parameters, const interfaces::Objects* objects = nullptr, interfaces::DepthMaps* depthMaps = nullptr);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase&) override;
    void updateDescriptors(const utils::BuffersDatabase&, const utils::AttachmentsDatabase&) override{};
};

}
#endif // SHADOW_H
