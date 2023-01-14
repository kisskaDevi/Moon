#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <libs/vulkan/vulkan.h>
#include "../attachments.h"

#include <string>
#include <unordered_map>

class                               texture;
class                               cubeTexture;
class                               object;
class                               camera;
class                               light;
struct                              Node;
struct                              Material;
struct                              MaterialBlock;
struct                              gltfModel;

class deferredGraphics
{
private:
    VkPhysicalDevice*               physicalDevice{nullptr};
    VkDevice*                       device{nullptr};
    VkQueue*                        graphicsQueue{nullptr};
    VkCommandPool*                  commandPool{nullptr};

    texture*                        emptyTexture{nullptr};
    uint32_t                        primitiveCount{0};
    bool                            transparencyPass{false};

    imageInfo                       image;

    std::vector<attachments*>       pAttachments;

    VkRenderPass                    renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>      framebuffers;

    struct Base{
        std::string                                     ExternalPath;

        std::unordered_map<uint8_t, VkPipelineLayout>   PipelineLayoutDictionary;
        std::unordered_map<uint8_t, VkPipeline>         PipelineDictionary;

        VkDescriptorSetLayout                           SceneDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout                           ObjectDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout                           PrimitiveDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout                           MaterialDescriptorSetLayout{VK_NULL_HANDLE};

        VkDescriptorPool                                DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>                    DescriptorSets;

        std::vector<VkBuffer>                           sceneUniformBuffers;
        std::vector<VkDeviceMemory>                     sceneUniformBuffersMemory;

        std::vector<object *>                           objects;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
        void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount);
            void renderNode(VkCommandBuffer commandBuffer, Node *node, VkPipelineLayout* pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t* primitiveCount);
    }base;

    struct OutliningExtension{
        std::string                     ExternalPath;

        Base*                           Parent{nullptr};

        VkPipelineLayout                outliningPipelineLayout{VK_NULL_HANDLE};
        VkPipeline                      outliningPipeline{VK_NULL_HANDLE};

        void DestroyOutliningPipeline(VkDevice* device);
        void createOutliningPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
            void renderNode(VkCommandBuffer commandBuffer, Node *node, VkPipelineLayout* pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t* primitiveCount);
    }outlining;

    struct Lighting{
        std::string                                         ExternalPath;
        bool                                                enableScattering{true};

        VkDescriptorSetLayout                               DescriptorSetLayout{VK_NULL_HANDLE};
        std::unordered_map<uint8_t, VkDescriptorSetLayout>  DescriptorSetLayoutDictionary;
        std::unordered_map<uint8_t, VkPipelineLayout>       PipelineLayoutDictionary;
        std::unordered_map<uint8_t, VkPipeline>             PipelinesDictionary;

        VkDescriptorPool                                    DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>                        DescriptorSets;

        std::vector<VkBuffer>                               uniformBuffers;
        std::vector<VkDeviceMemory>                         uniformBuffersMemory;

        std::vector<light*>                                 lightSources;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
            void createSpotPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass, std::string vertShaderPath, std::string fragShaderPath, VkPipelineLayout* pipelineLayout,VkPipeline* pipeline);
        void createDescriptorSetLayout(VkDevice* device);
        void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }lighting;

    struct AmbientLighting{
        std::string                                         ExternalPath;
        float                                               minAmbientFactor{0.05f};

        Lighting*                                           Parent{nullptr};

        VkPipelineLayout                                    PipelineLayout{VK_NULL_HANDLE};
        VkPipeline                                          Pipeline{VK_NULL_HANDLE};

        void DestroyPipeline(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }ambientLighting;

    struct Skybox
    {
        std::string                     ExternalPath;
        cubeTexture*                    texture{nullptr};

        VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
        VkPipeline                      Pipeline{VK_NULL_HANDLE};

        VkDescriptorSetLayout           DescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>    DescriptorSets;

        std::vector<VkBuffer>           uniformBuffers;
        std::vector<VkDeviceMemory>     uniformBuffersMemory;

        std::vector<object *>           objects;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
        void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }skybox;

public:
    deferredGraphics();
    void destroy();
    void destroyEmptyTexture();

    void setExternalPath(const std::string& path);
    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool);
    void setImageProp(imageInfo* pInfo);
    void setEmptyTexture(std::string ZERO_TEXTURE);

    void setMinAmbientFactor(const float& minAmbientFactor);
    void setScattering(const bool& enableScattering);
    void setTransparencyPass(const bool& transparencyPass);

    texture*  getEmptyTexture();
    VkBuffer* getSceneBuffer();

    void setAttachments(DeferredAttachments* pAttachments);
    void createAttachments(DeferredAttachments* pAttachments);

    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createBaseDescriptorPool();
    void createBaseDescriptorSets();
    void updateBaseDescriptorSets(attachments* depthAttachment, VkBuffer* storageBuffers);

    void createLightingDescriptorPool();
    void createLightingDescriptorSets();
    void updateLightingDescriptorSets();

    void createSkyboxDescriptorPool();
    void createSkyboxDescriptorSets();
    void updateSkyboxDescriptorSets();

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);

    void updateUniformBuffer(uint32_t currentImage, const camera& cameraObject);
    void updateSkyboxUniformBuffer(uint32_t currentImage, const camera& cameraObject);
    void updateObjectUniformBuffer(uint32_t currentImage);

    void bindBaseObject(object* newObject);
    void bindSkyBoxObject(object* newObject, const std::vector<std::string>& TEXTURE_PATH);

    bool removeBaseObject(object* object);
    bool removeSkyBoxObject(object* object);

    void addLightSource(light* lightSource);
    void removeLightSource(light* lightSource);

    void updateLightUbo(uint32_t imageIndex);
    void updateLightCmd(uint32_t imageIndex);
    void getLightCommandbuffers(std::vector<VkCommandBuffer>& commandbufferSet, uint32_t imageIndex);
};

#endif // GRAPHICS_H
