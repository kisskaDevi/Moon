#ifndef GRAPHICS_H
#define GRAPHICS_H

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>

#include <libs/glm/glm/glm.hpp>
#include <libs/glm/glm/gtc/matrix_transform.hpp>

#include <string>
#include <optional> // нужна для вызова std::optional<uint32_t>
#include "attachments.h"

class                               texture;
class                               cubeTexture;
class                               object;
class                               camera;
template <typename type> class      light;
class                               spotLight;
template <> class                   light<spotLight>;
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
};

struct lightPassPushConst
{
    alignas(4) float                minAmbientFactor;
};

struct postProcessingPushConst
{
    alignas(4) float                blitFactor;
};

struct SwapChainSupportDetails{
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

struct ShadowPassObjects{
    std::vector<object *> *base;
    std::vector<object *> *oneColor;
    std::vector<object *> *stencil;
};

struct QueueFamilyIndices
{
    std::optional<uint32_t>         graphicsFamily;                     //графикческое семейство очередей
    std::optional<uint32_t>         presentFamily;                      //семейство очередей показа
    bool isComplete()                                                   //если оба значения не пусты, а были записаны, выводит true
    {return graphicsFamily.has_value() && presentFamily.has_value();}
    //std::optional это оболочка, которая не содержит значения, пока вы ей что-то не присвоите.
    //В любой момент вы можете запросить, содержит ли он значение или нет, вызвав его has_value()функцию-член.
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

class graphics
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

    struct Second{
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

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
        void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount);

        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t lightSourcesCount,light<spotLight>** ppLightSources);
    }second;

    void createColorAttachments();
    void createDepthAttachment();
    void createResolveAttachments();

    void oneSampleRenderPass();
    void multiSampleRenderPass();
    void oneSampleFrameBuffer();
    void multiSampleFrameBuffer();

public:
    graphics();
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

    void createSecondDescriptorPool();
    void createSecondDescriptorSets();
    void updateSecondDescriptorSets();

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t lightSourceCount, light<spotLight>** pplightSources);

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

    void createLightDescriptorPool(light<spotLight>* object);
    void createLightDescriptorSets(light<spotLight>* object);
    void updateLightDescriptorSets(light<spotLight>* object);

    DeferredAttachments             getDeferredAttachments();
    std::vector<VkBuffer>&          getSceneBuffer();
    ShadowPassObjects               getObjects();
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
