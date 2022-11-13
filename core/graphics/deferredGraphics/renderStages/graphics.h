#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <libs/vulkan/vulkan.h>
#include <libs/glm/glm.hpp>
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
    VkPhysicalDevice*               physicalDevice;
    VkDevice*                       device;
    VkQueue*                        graphicsQueue;
    VkCommandPool*                  commandPool;

    texture*                        emptyTexture;
    camera*                         cameraObject;
    uint32_t                        primitiveCount = 0;
    bool                            transparencyPass = false;

    imageInfo                       image;

    std::vector<attachment>         colorAttachments;
    std::vector<attachments*>       Attachments;
    attachment*                     depthAttachment;

    VkRenderPass                    renderPass;
    std::vector<VkFramebuffer>      framebuffers;

    std::vector<VkBuffer>           storageBuffers;
    std::vector<VkDeviceMemory>     storageBuffersMemory;

    struct Base{
        std::string                     ExternalPath;

        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;

        VkDescriptorSetLayout           SceneDescriptorSetLayout;
        VkDescriptorSetLayout           ObjectDescriptorSetLayout;
        VkDescriptorSetLayout           PrimitiveDescriptorSetLayout;
        VkDescriptorSetLayout           MaterialDescriptorSetLayout;

        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;

        std::vector<VkBuffer>           sceneUniformBuffers;
        std::vector<VkDeviceMemory>     sceneUniformBuffersMemory;

        std::vector<object *>           objects;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
        void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount);
            void renderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount);
    }base;

    struct OutliningExtension{
        std::string                     ExternalPath;

        Base*                           base;

        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;

        VkPipelineLayout                outliningPipelineLayout;
        VkPipeline                      outliningPipeline;

        std::vector<object *>           objects;

        void DestroyPipeline(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void DestroyOutliningPipeline(VkDevice* device);
        void createOutliningPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount);
            void renderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount);
            void outliningRenderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets);
    }outlining;

    struct Skybox
    {
        std::string                     ExternalPath;
        cubeTexture*                    texture = nullptr;

        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;

        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorPool                DescriptorPool;
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

    struct Lighting{
        std::string                     ExternalPath;

        std::unordered_map<uint8_t, VkPipelineLayout>  PipelineLayoutDictionary;
        std::unordered_map<uint8_t, VkPipeline>        PipelinesDictionary;

        VkPipelineLayout                AmbientPipelineLayout;
        VkPipeline                      AmbientPipeline;

        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorSetLayout           LightDescriptorSetLayout;

        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;

        std::vector<VkBuffer>           uniformBuffers;
        std::vector<VkDeviceMemory>     uniformBuffersMemory;

        float                           minAmbientFactor = 0.05f;
        bool                            enableScattering = true;

        std::vector<light*>             lightSources;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
            void createAmbientPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
            void createSpotPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
            void createShadowSpotPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
            void createScatteringSpotPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
            void createScatteringShadowSpotPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
        void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }lighting;

    void createColorAttachments();
    void createDepthAttachment();
    void createResolveAttachments();

    void oneSampleRenderPass();
    void multiSampleRenderPass();
    void oneSampleFrameBuffer();
    void multiSampleFrameBuffer();

    void setAttachments(DeferredAttachments* Attachments);
public:
    deferredGraphics();
    void destroy();
    void destroyEmptyTexture();

    void setExternalPath(const std::string& path);
    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool);
    void setImageProp(imageInfo* pInfo);
    void setEmptyTexture(std::string ZERO_TEXTURE);
    void setCameraObject(camera* cameraObject);

    void setMinAmbientFactor(const float& minAmbientFactor);
    void setScattering(const bool& enableScattering);
    void setTransparencyPass(const bool& transparencyPass);

    texture*                getEmptyTexture();
    std::vector<VkBuffer>&  getSceneBuffer();

    void createAttachments(DeferredAttachments* Attachments);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createBaseDescriptorPool();
    void createBaseDescriptorSets();
    void updateBaseDescriptorSets(attachment* depthAttachment);

    void createSkyboxDescriptorPool();
    void createSkyboxDescriptorSets();
    void updateSkyboxDescriptorSets();

    void createLightingDescriptorPool();
    void createLightingDescriptorSets();
    void updateLightingDescriptorSets();

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);

    void updateUniformBuffer(uint32_t currentImage);
    void updateSkyboxUniformBuffer(uint32_t currentImage);
    void updateObjectUniformBuffer(uint32_t currentImage);

    void createStorageBuffers(uint32_t imageCount);
    void updateStorageBuffer(uint32_t currentImage, const glm::vec4& mousePosition);
    uint32_t readStorageBuffer(uint32_t currentImage);

    void bindBaseObject(object* newObject);
    void bindOutliningObject(object* newObject);
    void bindSkyBoxObject(object* newObject, const std::vector<std::string>& TEXTURE_PATH);

    bool removeBaseObject(object* object);
    bool removeOutliningObject(object* object);
    bool removeSkyBoxObject(object* object);
    void removeBinds();

    void addLightSource(light* lightSource);
    void removeLightSource(light* lightSource);

    void updateLightUbo(uint32_t imageIndex);
    void updateLightCmd(uint32_t imageIndex);
    void getLightCommandbuffers(std::vector<VkCommandBuffer>& commandbufferSet, uint32_t imageIndex);
};

#endif // GRAPHICS_H
