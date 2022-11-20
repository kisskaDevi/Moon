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
    VkPhysicalDevice*               physicalDevice{nullptr};
    VkDevice*                       device{nullptr};
    VkQueue*                        graphicsQueue{nullptr};
    VkCommandPool*                  commandPool{nullptr};

    texture*                        emptyTexture{nullptr};
    camera*                         cameraObject{nullptr};
    uint32_t                        primitiveCount{0};
    bool                            transparencyPass{false};

    imageInfo                       image;

    std::vector<attachment>         colorAttachments;
    std::vector<attachments*>       pAttachments;
    attachment*                     depthAttachment{nullptr};

    VkRenderPass                    renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>      framebuffers;

    std::vector<VkBuffer>           storageBuffers;
    std::vector<VkDeviceMemory>     storageBuffersMemory;

    struct Base{
        std::string                     ExternalPath;

        VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
        VkPipeline                      Pipeline{VK_NULL_HANDLE};

        VkDescriptorSetLayout           SceneDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout           ObjectDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout           PrimitiveDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout           MaterialDescriptorSetLayout{VK_NULL_HANDLE};

        VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
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

        Base*                           base{nullptr};

        VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
        VkPipeline                      Pipeline{VK_NULL_HANDLE};

        VkPipelineLayout                outliningPipelineLayout{VK_NULL_HANDLE};
        VkPipeline                      outliningPipeline{VK_NULL_HANDLE};

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

    struct Lighting{
        std::string                     ExternalPath;

        VkPipelineLayout                                    AmbientPipelineLayout{VK_NULL_HANDLE};
        VkPipeline                                          AmbientPipeline{VK_NULL_HANDLE};

        std::unordered_map<uint8_t, VkPipelineLayout>       PipelineLayoutDictionary;
        std::unordered_map<uint8_t, VkPipeline>             PipelinesDictionary;

        VkDescriptorSetLayout                               DescriptorSetLayout{VK_NULL_HANDLE};
        std::unordered_map<uint8_t, VkDescriptorSetLayout>  LightDescriptorSetLayout;

        VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>    DescriptorSets;

        std::vector<VkBuffer>           uniformBuffers;
        std::vector<VkDeviceMemory>     uniformBuffersMemory;

        float                           minAmbientFactor{0.05f};
        bool                            enableScattering{true};

        std::vector<light*>             lightSources;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
            void createAmbientPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
            void createSpotPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass, std::string vertShaderPath, std::string fragShaderPath, VkPipelineLayout* pipelineLayout,VkPipeline* pipeline);
        void createDescriptorSetLayout(VkDevice* device);
        void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }lighting;

    void oneSampleRenderPass();
    void multiSampleRenderPass();
    void oneSampleFrameBuffer();
    void multiSampleFrameBuffer();

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

    texture*  getEmptyTexture();
    VkBuffer* getSceneBuffer();

    void setAttachments(DeferredAttachments* pAttachments);
    void createAttachments(DeferredAttachments* pAttachments);
    void createBufferAttachments();

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

    void addLightSource(light* lightSource);
    void removeLightSource(light* lightSource);

    void updateLightUbo(uint32_t imageIndex);
    void updateLightCmd(uint32_t imageIndex);
    void getLightCommandbuffers(std::vector<VkCommandBuffer>& commandbufferSet, uint32_t imageIndex);
};

#endif // GRAPHICS_H
