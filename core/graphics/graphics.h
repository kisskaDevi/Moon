#ifndef GRAPHICS_H
#define GRAPHICS_H

#define GLFW_INCLUDE_VULKAN
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>

#include <libs/glm/glm/glm.hpp>
#include <libs/glm/glm/gtc/matrix_transform.hpp>

#include <string>
#include <iostream>
#include "attachments.h"
#include "core/operations.h"

class                               texture;
class                               cubeTexture;
class                               object;
class                               camera;
class                               spotLight;
struct                              Node;
struct                              Material;
struct                              MaterialBlock;
struct                              gltfModel;

struct UniformBufferObject{
    alignas(16) glm::mat4           view;
    alignas(16) glm::mat4           proj;
    alignas(16) glm::vec4           eyePosition;
};

struct SkyboxUniformBufferObject{
    alignas(16) glm::mat4           proj;
    alignas(16) glm::mat4           view;
    alignas(16) glm::mat4           model;
};

struct StorageBufferObject{
    alignas(16) glm::vec4           mousePosition;
    alignas(4)  int                 number;
    alignas(4)  float               depth;
};

struct StencilPushConst
{
    alignas(16) glm::vec4           stencilColor;
    alignas(4)  float               width;
};

struct lightPassPushConst
{
    alignas(4) float                minAmbientFactor;
};

struct postProcessingPushConst
{
    alignas(4) float                blitFactor;
};

struct GBufferAttachments{
    attachments*        position;
    attachments*        normal;
    attachments*        color;
    attachments*        emission;
};

struct DeferredAttachments{
    attachments*        image;
    attachments*        blur;
    attachments*        bloom;
    GBufferAttachments  GBuffer;
};

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

    imageInfo                       image;

    std::vector<attachment>         colorAttachments;
    std::vector<attachments>        Attachments;
    attachment                      depthAttachment;

    VkRenderPass                    renderPass;
    std::vector<VkFramebuffer>      framebuffers;

    std::vector<VkBuffer>           storageBuffers;
    std::vector<VkDeviceMemory>     storageBuffersMemory;

    struct Base{
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

        void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount);
        void createDescriptorSetLayout(VkDevice* device);
        void createObjectDescriptorPool(VkDevice* device, object* object, uint32_t imageCount);
        void createObjectDescriptorSet(VkDevice* device, object* object, uint32_t imageCount);

        void createModelDescriptorPool(VkDevice* device, gltfModel* pModel);
        void createModelDescriptorSet(VkDevice* device, gltfModel* pModel, texture* emptyTexture);
            void createModelNodeDescriptorSet(VkDevice* device, gltfModel* pModel, Node* node);
            void createModelMaterialDescriptorSet(VkDevice* device, gltfModel* pModel, Material* material, texture* emptyTexture);

        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount);
            void renderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount);
    }base;

    struct bloomExtension{
        Base                            *base;
        VkPipeline                      Pipeline;
        VkPipelineLayout                PipelineLayout;

        std::vector<object *>           objects;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount);
            void renderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount);
    }bloom;

    struct oneColorExtension{
        Base                            *base;
        VkPipeline                      Pipeline;
        VkPipelineLayout                PipelineLayout;

        std::vector<object *>           objects;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount);
            void renderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount);
    }oneColor;

    struct StencilExtension{
        Base                            *base;
        VkPipeline                      firstPipeline;
        VkPipeline                      secondPipeline;
        VkPipelineLayout                firstPipelineLayout;
        VkPipelineLayout                secondPipelineLayout;

        std::vector<object *>           objects;

        void DestroyFirstPipeline(VkDevice* device);
        void DestroySecondPipeline(VkDevice* device);
        void createFirstPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createSecondPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount);
            void renderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount);
            void stencilRenderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets);
    }stencil;

    struct Skybox
    {
        cubeTexture                     *texture = nullptr;
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

    struct SpotLighting{
        VkPipelineLayout                PipelineLayout;
        VkPipelineLayout                AmbientPipelineLayout;
        VkPipeline                      Pipeline;
        VkPipeline                      ScatteringPipeline;
        VkPipeline                      AmbientPipeline;
        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorSetLayout           LightDescriptorSetLayout;
        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;
        std::vector<VkBuffer>           uniformBuffers;
        std::vector<VkDeviceMemory>     uniformBuffersMemory;

        float                           minAmbientFactor = 0.05f;

        std::vector<spotLight*>                     lightSources;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
        void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount);

        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }spotLighting;

    void createColorAttachments();
    void createDepthAttachment();
    void createResolveAttachments();

    void oneSampleRenderPass();
    void multiSampleRenderPass();
    void oneSampleFrameBuffer();
    void multiSampleFrameBuffer();

public:
    deferredGraphics();
    void destroy();
    void destroyEmptyTexture();

    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool);
    void setImageProp(imageInfo* pInfo);
    void setEmptyTexture(std::string ZERO_TEXTURE);
    void setCameraObject(camera* cameraObject);

    void setMinAmbientFactor(const float& minAmbientFactor);

    void createAttachments();
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createBaseDescriptorPool();
    void createBaseDescriptorSets();
    void updateBaseDescriptorSets();

    void createSkyboxDescriptorPool();
    void createSkyboxDescriptorSets();
    void updateSkyboxDescriptorSets();

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);

    void updateUniformBuffer(uint32_t currentImage);
    void updateSkyboxUniformBuffer(uint32_t currentImage);
    void updateObjectUniformBuffer(uint32_t currentImage);

    void createStorageBuffers(uint32_t imageCount);
    void updateStorageBuffer(uint32_t currentImage, const glm::vec4& mousePosition);
    uint32_t readStorageBuffer(uint32_t currentImage);

    void createModel(gltfModel* pModel);
    void destroyModel(gltfModel* pModel);

    void bindBaseObject(object* newObject);
    void bindBloomObject(object* newObject);
    void bindOneColorObject(object* newObject);
    void bindStencilObject(object* newObject, float lineWidth, glm::vec4 lineColor);
    void bindSkyBoxObject(object* newObject, const std::vector<std::string>& TEXTURE_PATH);

    bool removeBaseObject(object* object);
    bool removeBloomObject(object* object);
    bool removeOneColorObject(object* object);
    bool removeStencilObject(object* object);
    bool removeSkyBoxObject(object* object);
    void removeBinds();

    void addSpotLightSource(spotLight* lightSource, QueueFamilyIndices* queueFamilyIndices);
    void removeSpotLightSource(spotLight* lightSource);
    void createSpotLightDescriptorPool(spotLight* object);
    void createSpotLightDescriptorSets(spotLight* object);
    void updateSpotLightDescriptorSets(spotLight* object);

    void createSpotLightingDescriptorPool();
    void createSpotLightingDescriptorSets();
    void updateSpotLightingDescriptorSets();

    void updateSpotLightUbo(uint32_t imageIndex);
    void updateSpotLightCmd(uint32_t imageIndex);
    void getSpotLightCommandbuffers(std::vector<VkCommandBuffer>* commandbufferSet, uint32_t imageIndex);

    DeferredAttachments             getDeferredAttachments();
    std::vector<VkBuffer>&          getSceneBuffer();
};

class postProcessing
{
private:
    VkPhysicalDevice*                   physicalDevice;
    VkDevice*                           device;
    VkQueue*                            graphicsQueue;
    VkCommandPool*                      commandPool;
    QueueFamilyIndices*                 queueFamilyIndices;
    VkSurfaceKHR*                       surface;

    imageInfo                           image;

    VkSwapchainKHR                      swapChain;
    uint32_t                            swapChainAttachmentCount = 1;
    std::vector<attachments>            swapChainAttachments;

    uint32_t                            AttachmentCount = 1;
    std::vector<attachments>            Attachments;

    float                               blitFactor = 1.5f;
    static const uint32_t               blitAttachmentCount = 8;
    std::vector<attachments>            blitAttachments;
    attachments                         blitAttachment;

    attachments                         sslrAttachment;
    attachments                         ssaoAttachment;

    VkRenderPass                        renderPass;
    std::vector<VkFramebuffer>          framebuffers;

    struct First{
        VkPipelineLayout                    PipelineLayout;
        VkPipeline                          Pipeline;
        VkDescriptorSetLayout               DescriptorSetLayout;
        VkDescriptorPool                    DescriptorPool;
        std::vector<VkDescriptorSet>        DescriptorSets;
    }first;

    struct Second{
        VkPipelineLayout                    PipelineLayout;
        VkPipeline                          Pipeline;
        VkDescriptorSetLayout               DescriptorSetLayout;
        VkDescriptorPool                    DescriptorPool;
        std::vector<VkDescriptorSet>        DescriptorSets;
    }second;

    //Создание цепочки обмена
    void createSwapChain(GLFWwindow* window, SwapChainSupportDetails swapChainSupport);
    void createImageViews();
    void createColorAttachments();
public:
    postProcessing();
    void destroy();
    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool, QueueFamilyIndices* queueFamilyIndices, VkSurfaceKHR* surface);
    void setImageProp(imageInfo* pInfo);

    void  setBlitFactor(const float& blitFactor);
    float getBlitFactor();

    void createAttachments(GLFWwindow* window, SwapChainSupportDetails swapChainSupport);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
        void createDescriptorSetLayout();
        void createFirstGraphicsPipeline();
        void createSecondGraphicsPipeline();

    void createDescriptorPool();
    void createDescriptorSets(DeferredAttachments Attachments);

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);

    std::vector<attachments>        & getBlitAttachments();
    attachments                     & getBlitAttachment();
    attachments                     & getSSLRAttachment();
    attachments                     & getSSAOAttachment();

    VkSwapchainKHR                  & SwapChain();
    VkFormat                        & SwapChainImageFormat();
    VkExtent2D                      & SwapChainImageExtent();
    uint32_t                        & SwapChainImageCount();
};

#endif // GRAPHICS_H
