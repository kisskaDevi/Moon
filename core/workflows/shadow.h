#ifndef SHADOW_H
#define SHADOW_H

#include "workflow.h"
#include "vkdefault.h"
#include <unordered_map>

namespace moon::interfaces {
class Object;
class Light;
}
namespace moon::utils { class DepthMap;}

namespace moon::workflows {

class ShadowGraphics : public Workflow
{
private:
    std::unordered_map<const moon::utils::DepthMap*, utils::vkDefault::Framebuffers> framebuffersMap;
    bool enable{true};

    struct Shadow : public Workbody{
        Shadow(const moon::utils::ImageInfo& imageInfo) : Workbody(imageInfo) {};

        void create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) override;

        moon::utils::vkDefault::DescriptorSetLayout lightUniformBufferSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout objectDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout primitiveDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout materialDescriptorSetLayout;

        std::vector<moon::interfaces::Object*>* objects{ nullptr };
        std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap>* depthMaps{ nullptr };
    }shadow;

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, moon::interfaces::Light* lightSource, const moon::utils::DepthMap& depthMap);
    void createRenderPass();
public:
    ShadowGraphics(const moon::utils::ImageInfo& imageInfo,
                   const std::filesystem::path& shadersPath,
                   bool enable,
                   std::vector<moon::interfaces::Object*>* objects = nullptr,
                   std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap>* depthMaps = nullptr);

    void create(moon::utils::AttachmentsDatabase&) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase&, const moon::utils::AttachmentsDatabase&) override{};
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // SHADOW_H
