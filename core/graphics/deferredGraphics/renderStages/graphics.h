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

    texture*                        emptyTexture{nullptr};

    uint32_t                        primitiveCount{0};

    imageInfo                       image;

    std::vector<attachments*>       pAttachments;

    VkRenderPass                    renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>      framebuffers;

    std::vector<VkCommandBuffer>          commandBuffers;

    struct Base{
        std::string                                     ExternalPath;
        bool                                            transparencyPass{false};

        std::unordered_map<uint8_t, VkPipelineLayout>   PipelineLayoutDictionary;
        std::unordered_map<uint8_t, VkPipeline>         PipelineDictionary;

        VkDescriptorSetLayout                           SceneDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout                           ObjectDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout                           PrimitiveDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout                           MaterialDescriptorSetLayout{VK_NULL_HANDLE};

        VkDescriptorPool                                DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>                    DescriptorSets;

        std::vector<object *>                           objects;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
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

        std::vector<light*>                                 lightSources;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
            void createSpotPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass, std::string vertShaderPath, std::string fragShaderPath, VkPipelineLayout* pipelineLayout,VkPipeline* pipeline);
        void createDescriptorSetLayout(VkDevice* device);
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

    void createBaseDescriptorPool();
    void createBaseDescriptorSets();
    void updateBaseDescriptorSets(attachments* depthAttachment, VkBuffer* storageBuffers, size_t sizeOfStorageBuffers, camera* cameraObject);

    void createLightingDescriptorPool();
    void createLightingDescriptorSets();
    void updateLightingDescriptorSets(camera* cameraObject);
public:
    deferredGraphics();
    void destroy();
    void freeCommandBuffer(VkCommandPool commandPool){
        if(commandBuffers.data()){
            vkFreeCommandBuffers(*device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
        }
        commandBuffers.resize(0);
    }

    void setEmptyTexture(texture* emptyTexture);
    void setExternalPath(const std::string& path);
    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device);
    void setImageProp(imageInfo* pInfo);

    void setAttachments(DeferredAttachments* pAttachments);
    void createAttachments(DeferredAttachments* pAttachments);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createDescriptorPool();
    void createDescriptorSets();
    void updateDescriptorSets(attachments* depthAttachment, VkBuffer* storageBuffers, size_t sizeOfStorageBuffer, camera* cameraObject);

    void beginCommandBuffer(uint32_t frameNumber);
    void endCommandBuffer(uint32_t frameNumber);

    void createCommandBuffers(VkCommandPool commandPool);
    void updateCommandBuffer(uint32_t frameNumber);
    VkCommandBuffer& getCommandBuffer(uint32_t frameNumber);

    void updateObjectUniformBuffer(VkCommandBuffer commandBuffer, uint32_t currentImage);
    void updateLightSourcesUniformBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

    void bindBaseObject(object* newObject);
    bool removeBaseObject(object* object);

    void addLightSource(light* lightSource);
    void removeLightSource(light* lightSource);

    void setMinAmbientFactor(const float& minAmbientFactor);
    void setScattering(const bool& enableScattering);
    void setTransparencyPass(const bool& transparencyPass);
};

#endif // GRAPHICS_H
