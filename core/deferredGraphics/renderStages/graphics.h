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
        std::filesystem::path       shadersPath;
        bool                        transparencyPass{false};
        bool                        enableTransparency{false};
        uint32_t                    transparencyNumber{0};
        const utils::ImageInfo&     imageInfo;
        const GraphicsParameters&   parameters;

        VkDevice device{VK_NULL_HANDLE};

        moon::utils::vkDefault::PipelineLayoutMap   pipelineLayoutMap;
        moon::utils::vkDefault::PipelineMap         pipelineMap;

        moon::utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout objectDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout primitiveDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout materialDescriptorSetLayout;

        moon::utils::vkDefault::DescriptorPool descriptorPool;
        moon::utils::vkDefault::DescriptorSets descriptorSets;

        std::vector<moon::interfaces::Object*>* objects{nullptr};

        Base(const bool transparencyPass,
             const bool enableTransparency,
             const uint32_t transparencyNumber,
             const utils::ImageInfo& imageInfo,
             const GraphicsParameters& parameters);

        void createPipeline(VkRenderPass pRenderPass);
        void createDescriptorSetLayout();
        void createDescriptors();
        void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase);

        void create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount) const;
    }base;

    struct OutliningExtension{
        std::filesystem::path   shadersPath;
        const Base&             parent;

        VkDevice device{ VK_NULL_HANDLE };

        moon::utils::vkDefault::PipelineLayout  pipelineLayout;
        moon::utils::vkDefault::Pipeline        pipeline;

        OutliningExtension(const Base& parent);

        void create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const;
    }outlining;

    struct Lighting{
        std::filesystem::path       shadersPath;
        bool                        enableScattering{true};
        const utils::ImageInfo&     imageInfo;
        const GraphicsParameters&   parameters;

        VkDevice device{ VK_NULL_HANDLE };

        moon::utils::vkDefault::DescriptorSetLayout     descriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout     shadowDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayoutMap  bufferDescriptorSetLayoutMap;
        moon::utils::vkDefault::DescriptorSetLayoutMap  textureDescriptorSetLayoutMap;

        moon::utils::vkDefault::PipelineLayoutMap   pipelineLayoutMap;
        moon::utils::vkDefault::PipelineMap         pipelineMap;

        moon::utils::vkDefault::DescriptorPool descriptorPool;
        moon::utils::vkDefault::DescriptorSets descriptorSets;

        std::vector<moon::interfaces::Light*>* lightSources;
        std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>* depthMaps;

        Lighting(const utils::ImageInfo& imageInfo, const GraphicsParameters& parameters);

        void createPipeline(VkRenderPass pRenderPass);
        void createPipeline(uint8_t mask, VkRenderPass pRenderPass, std::filesystem::path vertShadersPath, std::filesystem::path fragShadersPath);
        void createDescriptorSetLayout();
        void createDescriptors();
        void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase);

        void create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffer) const;
    }lighting;

    struct AmbientLighting{
        std::filesystem::path   shadersPath;
        float                   minAmbientFactor{0.05f};
        const Lighting&         parent;

        VkDevice device{ VK_NULL_HANDLE };

        moon::utils::vkDefault::PipelineLayout  pipelineLayout;
        moon::utils::vkDefault::Pipeline        pipeline;

        AmbientLighting(const Lighting& parent);

        void create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const;
    }ambientLighting;

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

public:
    Graphics(const moon::utils::ImageInfo& imageInfo,
             const std::filesystem::path& shadersPath,
             GraphicsParameters parameters,
             bool enable,
             bool enableTransparency,
             bool transparencyPass,
             uint32_t transparencyNumber,
             std::vector<moon::interfaces::Object*>* object = nullptr,
             std::vector<moon::interfaces::Light*>* lightSources = nullptr,
             std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>* depthMaps = nullptr);

    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;

    Graphics& setMinAmbientFactor(const float& minAmbientFactor);
};

}
#endif // GRAPHICS_H
