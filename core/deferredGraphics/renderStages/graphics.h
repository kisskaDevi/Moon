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

struct GraphicsParameters : workflows::Parameters {
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

    bool        transparencyPass{ false };
    bool        enableTransparency{ false };
    uint32_t    transparencyNumber{ 0 };
    float       minAmbientFactor{ 0.05f };
};

class Graphics : public moon::workflows::Workflow
{
private:
    GraphicsParameters& parameters;
    DeferredAttachments deferredAttachments;

    struct Base{
        std::filesystem::path       shadersPath;
        const utils::ImageInfo&     imageInfo;
        const GraphicsParameters&   parameters;

        moon::utils::vkDefault::PipelineLayoutMap   pipelineLayoutMap;
        moon::utils::vkDefault::PipelineMap         pipelineMap;

        moon::utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout objectDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout primitiveDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout materialDescriptorSetLayout;

        moon::utils::vkDefault::DescriptorPool descriptorPool;
        moon::utils::vkDefault::DescriptorSets descriptorSets;

        std::vector<moon::interfaces::Object*>* objects{nullptr};

        Base(const utils::ImageInfo& imageInfo, const GraphicsParameters& parameters);

        void createPipeline(VkDevice device, VkRenderPass pRenderPass);
        void createDescriptorSetLayout(VkDevice device);
        void createDescriptors(VkDevice device);
        void updateDescriptorSets(VkDevice device, const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase);

        void create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount) const;
    }base;

    struct OutliningExtension{
        std::filesystem::path   shadersPath;
        const Base&             parent;

        moon::utils::vkDefault::PipelineLayout  pipelineLayout;
        moon::utils::vkDefault::Pipeline        pipeline;

        OutliningExtension(const Base& parent);

        void create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const;
    }outlining;

    struct Lighting{
        std::filesystem::path       shadersPath;
        const utils::ImageInfo&     imageInfo;
        const GraphicsParameters&   parameters;

        moon::utils::vkDefault::DescriptorSetLayout     descriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayout     shadowDescriptorSetLayout;
        moon::utils::vkDefault::DescriptorSetLayoutMap  bufferDescriptorSetLayoutMap;
        moon::utils::vkDefault::DescriptorSetLayoutMap  textureDescriptorSetLayoutMap;

        moon::utils::vkDefault::PipelineLayoutMap   pipelineLayoutMap;
        moon::utils::vkDefault::PipelineMap         pipelineMap;

        moon::utils::vkDefault::DescriptorPool descriptorPool;
        moon::utils::vkDefault::DescriptorSets descriptorSets;

        std::vector<moon::interfaces::Light*>* lightSources;
        std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap>* depthMaps;

        Lighting(const utils::ImageInfo& imageInfo, const GraphicsParameters& parameters);

        void createPipeline(VkDevice device, VkRenderPass pRenderPass);
        void createPipeline(VkDevice device, uint8_t mask, VkRenderPass pRenderPass, std::filesystem::path vertShadersPath, std::filesystem::path fragShadersPath);
        void createDescriptorSetLayout(VkDevice device);
        void createDescriptors(VkDevice device);
        void updateDescriptorSets(VkDevice device, const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase);

        void create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffer) const;
    }lighting;

    struct AmbientLighting{
        std::filesystem::path   shadersPath;
        const Lighting&         parent;

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
             GraphicsParameters& parameters,
             std::vector<moon::interfaces::Object*>* object = nullptr,
             std::vector<moon::interfaces::Light*>* lightSources = nullptr,
             std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap>* depthMaps = nullptr);

    void create(moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

}
#endif // GRAPHICS_H
