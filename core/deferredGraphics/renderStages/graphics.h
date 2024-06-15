#ifndef GRAPHICS_H
#define GRAPHICS_H

#include "vkdefault.h"
#include "workflow.h"
#include "deferredAttachments.h"

#include <filesystem>
#include <unordered_map>

namespace moon::interfaces {
class   Object;
class   Light;
}
namespace moon::utils {
class   DepthMap;
class   Texture;
class   CubeTexture;
}

namespace moon::deferredGraphics {

struct GraphicsParameters{
    struct{
        std::string camera;
    }in;
    struct{
        std::string image;
        std::string blur;
        std::string bloom;
        std::string position;
        std::string normal;
        std::string color;
        std::string depth;
        std::string transparency;
    }out;
};

class Graphics : public moon::workflows::Workflow
{
private:
    GraphicsParameters parameters;

    uint32_t                        primitiveCount{0};
    DeferredAttachments             deferredAttachments;
    bool                            enable{true};

    struct Base{
        std::filesystem::path   shadersPath;
        bool                    transparencyPass{false};
        bool                    enableTransparency{false};
        uint32_t                transparencyNumber{0};

        moon::utils::vkDefault::PipelineLayoutMap   pipelineLayoutMap;
        moon::utils::vkDefault::PipelineMap         pipelineMap;

        moon::utils::vkDefault::DescriptorSetLayout baseDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout objectDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout primitiveDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout materialDescriptorSetLayout;

        VkDescriptorPool                                DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>                    DescriptorSets;

        std::vector<moon::interfaces::Object*>*         objects{nullptr};

        void destroy(VkDevice device);
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass);
        void createDescriptorSetLayout(VkDevice device);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount) const;
    }base;

    struct OutliningExtension{
        std::filesystem::path   shadersPath;
        Base*                   parent{nullptr};

        moon::utils::vkDefault::PipelineLayout  pipelineLayout;
        moon::utils::vkDefault::Pipeline        pipeline;

        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const;
    }outlining;

    struct Lighting{
        std::filesystem::path   shadersPath;
        bool                    enableScattering{true};

        moon::utils::vkDefault::DescriptorSetLayout     lightingDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout     shadowDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayoutMap  bufferDescriptorSetLayoutMap;
        moon::utils::vkDefault::DescriptorSetLayoutMap  textureDescriptorSetLayoutMap;

        moon::utils::vkDefault::PipelineLayoutMap   pipelineLayoutMap;
        moon::utils::vkDefault::PipelineMap         pipelineMap;

        VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>    DescriptorSets;

        std::vector<moon::interfaces::Light*>*                                lightSources;
        std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>* depthMaps;

        void destroy(VkDevice device);
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass);
        void createPipeline(uint8_t mask, VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass, std::filesystem::path vertShadersPath, std::filesystem::path fragShadersPath);
        void createDescriptorSetLayout(VkDevice device);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffer) const;
    }lighting;

    struct AmbientLighting{
        std::filesystem::path   shadersPath;
        float                   minAmbientFactor{0.05f};
        Lighting*               parent{nullptr};

        moon::utils::vkDefault::PipelineLayout  pipelineLayout;
        moon::utils::vkDefault::Pipeline        pipeline;

        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const;
    }ambientLighting;

    void createBaseDescriptorPool();
    void createBaseDescriptorSets();
    void updateBaseDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase);

    void createLightingDescriptorPool();
    void createLightingDescriptorSets();
    void updateLightingDescriptorSets(const moon::utils::BuffersDatabase& bDatabase);

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    Graphics(GraphicsParameters parameters,
             bool enable,
             bool enableTransparency,
             bool transparencyPass,
             uint32_t transparencyNumber,
             std::vector<moon::interfaces::Object*>* object = nullptr,
             std::vector<moon::interfaces::Light*>* lightSources = nullptr,
             std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>* depthMaps = nullptr);

    void destroy()override;
    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;

    Graphics& setMinAmbientFactor(const float& minAmbientFactor);
};

}
#endif // GRAPHICS_H
