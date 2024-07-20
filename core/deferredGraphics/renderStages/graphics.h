#ifndef GRAPHICS_H
#define GRAPHICS_H

#include "vkdefault.h"
#include "workflow.h"
#include "deferredAttachments.h"

#include "object.h"
#include "light.h"
#include "texture.h"

#include <filesystem>
#include <unordered_map>

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

class Graphics : public workflows::Workflow
{
private:
    GraphicsParameters& parameters;
    DeferredAttachments deferredAttachments;

    struct Base{
        std::filesystem::path       shadersPath;
        const utils::ImageInfo&     imageInfo;
        const GraphicsParameters&   parameters;

        utils::vkDefault::PipelineLayoutMap   pipelineLayoutMap;
        utils::vkDefault::PipelineMap         pipelineMap;

        utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout objectDescriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout primitiveDescriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout materialDescriptorSetLayout;

        utils::vkDefault::DescriptorPool descriptorPool;
        utils::vkDefault::DescriptorSets descriptorSets;

        const interfaces::Objects* objects{ nullptr };

        Base(const utils::ImageInfo& imageInfo, const GraphicsParameters& parameters, const interfaces::Objects* objects);

        void createPipeline(VkDevice device, VkRenderPass pRenderPass);
        void createDescriptorSetLayout(VkDevice device);
        void createDescriptors(VkDevice device);
        void updateDescriptorSets(VkDevice device, const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase);

        void create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount) const;
    }base;

    struct OutliningExtension{
        std::filesystem::path   shadersPath;
        const Base&             parent;

        utils::vkDefault::PipelineLayout  pipelineLayout;
        utils::vkDefault::Pipeline        pipeline;

        OutliningExtension(const Base& parent);

        void create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const;
    }outlining;

    struct Lighting{
        std::filesystem::path       shadersPath;
        const utils::ImageInfo&     imageInfo;
        const GraphicsParameters&   parameters;

        utils::vkDefault::DescriptorSetLayout     descriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout     shadowDescriptorSetLayout;
        utils::vkDefault::DescriptorSetLayoutMap  bufferDescriptorSetLayoutMap;
        utils::vkDefault::DescriptorSetLayoutMap  textureDescriptorSetLayoutMap;

        utils::vkDefault::PipelineLayoutMap   pipelineLayoutMap;
        utils::vkDefault::PipelineMap         pipelineMap;

        utils::vkDefault::DescriptorPool descriptorPool;
        utils::vkDefault::DescriptorSets descriptorSets;

        const interfaces::Lights* lightSources{ nullptr };
        const interfaces::DepthMaps* depthMaps{ nullptr };

        Lighting(const utils::ImageInfo& imageInfo, const GraphicsParameters& parameters, const interfaces::Lights* lightSources, const interfaces::DepthMaps* depthMaps);

        void createPipeline(VkDevice device, VkRenderPass pRenderPass);
        void createPipeline(VkDevice device, uint8_t mask, VkRenderPass pRenderPass, std::filesystem::path vertShadersPath, std::filesystem::path fragShadersPath);
        void createDescriptorSetLayout(VkDevice device);
        void createDescriptors(VkDevice device);
        void updateDescriptorSets(VkDevice device, const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase);

        void create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffer) const;
    }lighting;

    struct AmbientLighting{
        std::filesystem::path   shadersPath;
        const Lighting&         parent;

        utils::vkDefault::PipelineLayout  pipelineLayout;
        utils::vkDefault::Pipeline        pipeline;

        AmbientLighting(const Lighting& parent);

        void create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const;
    }ambientLighting;

    void createAttachments(utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    Graphics(GraphicsParameters& parameters,
             const interfaces::Objects* object = nullptr,
             const interfaces::Lights* lightSources = nullptr,
             const interfaces::DepthMaps* depthMaps = nullptr);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

}
#endif // GRAPHICS_H
