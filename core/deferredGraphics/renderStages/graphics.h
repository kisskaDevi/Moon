#ifndef GRAPHICS_H
#define GRAPHICS_H

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

struct graphicsParameters{
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

class graphics : public moon::workflows::Workflow
{
private:
    graphicsParameters parameters;

    uint32_t                        primitiveCount{0};
    DeferredAttachments             deferredAttachments;
    bool                            enable{true};

    struct Base{
        std::filesystem::path                           ShadersPath;
        bool                                            transparencyPass{false};
        bool                                            enableTransparency{false};
        uint32_t                                        transparencyNumber{0};

        std::unordered_map<uint8_t, VkPipelineLayout>   PipelineLayoutDictionary;
        std::unordered_map<uint8_t, VkPipeline>         PipelineDictionary;

        VkDescriptorSetLayout                           SceneDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout                           ObjectDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout                           PrimitiveDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout                           MaterialDescriptorSetLayout{VK_NULL_HANDLE};

        VkDescriptorPool                                DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>                    DescriptorSets;

        std::vector<moon::interfaces::Object*>*                           objects{nullptr};

        void Destroy(VkDevice device);
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass);
        void createDescriptorSetLayout(VkDevice device);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount);
    }base;

    struct OutliningExtension{
        std::filesystem::path           ShadersPath;

        Base*                           Parent{nullptr};

        VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
        VkPipeline                      Pipeline{VK_NULL_HANDLE};

        void DestroyPipeline(VkDevice device);
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }outlining;

    struct Lighting{
        std::filesystem::path                               ShadersPath;
        bool                                                enableScattering{true};

        VkDescriptorSetLayout                               DescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout                               ShadowDescriptorSetLayout{VK_NULL_HANDLE};
        std::unordered_map<uint8_t, VkDescriptorSetLayout>  BufferDescriptorSetLayoutDictionary;
        std::unordered_map<uint8_t, VkDescriptorSetLayout>  DescriptorSetLayoutDictionary;
        std::unordered_map<uint8_t, VkPipelineLayout>       PipelineLayoutDictionary;
        std::unordered_map<uint8_t, VkPipeline>             PipelinesDictionary;

        VkDescriptorPool                                    DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>                        DescriptorSets;

        std::vector<moon::interfaces::Light*>*                                lightSources;
        std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>* depthMaps;

        void Destroy(VkDevice device);
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass);
        void createPipeline(uint8_t mask, VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass, std::filesystem::path vertShadersPath, std::filesystem::path fragShadersPath);
        void createDescriptorSetLayout(VkDevice device);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffer);
    }lighting;

    struct AmbientLighting{
        std::filesystem::path                               ShadersPath;
        float                                               minAmbientFactor{0.05f};

        Lighting*                                           Parent{nullptr};

        VkPipelineLayout                                    PipelineLayout{VK_NULL_HANDLE};
        VkPipeline                                          Pipeline{VK_NULL_HANDLE};

        void DestroyPipeline(VkDevice device);
        void createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }ambientLighting;

    void createBaseDescriptorPool();
    void createBaseDescriptorSets();
    void updateBaseDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase& aDatabase);

    void createLightingDescriptorPool();
    void createLightingDescriptorSets();
    void updateLightingDescriptorSets(const moon::utils::BuffersDatabase& bDatabase);

    void setAttachments();

    void createAttachments(moon::utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    graphics(graphicsParameters parameters,
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

    graphics& setMinAmbientFactor(const float& minAmbientFactor);
};

#endif // GRAPHICS_H
