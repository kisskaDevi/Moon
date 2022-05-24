#ifndef GRAPHICS_H
#define GRAPHICS_H

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>

#include <libs/glm/glm/glm.hpp>
#include <libs/glm/glm/gtc/matrix_transform.hpp>

#include <optional> // нужна для вызова std::optional<uint32_t>
#include "attachments.h"

class                               VkApplication;
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

struct StencilPushConst
{
    alignas(16) glm::vec4           stencilColor;
};

class graphics
{
private:
    VkApplication                   *app;
    texture                         *emptyTexture;
    camera                          *cameraObject;
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

        void Destroy(VkApplication  *app);
        void createPipeline(VkApplication* app, imageInfo* pInfo, VkRenderPass* pRenderPass);

        void createDescriptorSetLayout(VkApplication *app);
        void createObjectDescriptorPool(VkApplication *app, object *object, uint32_t imageCount);
        void createObjectDescriptorSet(VkApplication *app, object *object, uint32_t imageCount, texture* emptyTexture);
            void createObjectNodeDescriptorSet(VkApplication *app, object *object, Node* node);
            void createObjectMaterialDescriptorSet(VkApplication *app, object *object, Material* material, texture* emptyTexture);
        void createUniformBuffers(VkApplication *app, uint32_t imageCount);

        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount);
            void renderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount);
    }base;

    struct bloomExtension{
        Base                            *base;
        VkPipeline                      Pipeline;
        VkPipelineLayout                PipelineLayout;

        std::vector<object *>           objects;

        void Destroy(VkApplication  *app);
        void createPipeline(VkApplication* app, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount);
            void renderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount);
    }bloom;

    struct oneColorExtension{
        Base                            *base;
        VkPipeline                      Pipeline;
        VkPipelineLayout                PipelineLayout;

        std::vector<object *>           objects;

        void Destroy(VkApplication  *app);
        void createPipeline(VkApplication* app, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount);
            void renderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount);
    }oneColor;

    struct StencilExtension{
        Base                            *base;
        VkPipeline                      firstPipeline;
        VkPipeline                      secondPipeline;
        VkPipelineLayout                firstPipelineLayout;
        VkPipelineLayout                secondPipelineLayout;

        std::vector<bool>               stencilEnable;
        std::vector<float>              stencilWidth;
        std::vector<glm::vec4>          stencilColor;
        std::vector<object *>           objects;

        void DestroyFirstPipeline(VkApplication *app);
        void DestroySecondPipeline(VkApplication *app);
        void createFirstPipeline(VkApplication* app, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createSecondPipeline(VkApplication* app, imageInfo* pInfo, VkRenderPass* pRenderPass);
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

        void Destroy(VkApplication* app);
        void createPipeline(VkApplication* app, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkApplication* app);
        void createUniformBuffers(VkApplication* app, uint32_t imageCount);
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

        void Destroy(VkApplication  *app);
        void createPipeline(VkApplication *app, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkApplication *app);
        void createUniformBuffers(VkApplication *app, uint32_t imageCount);

        void createLightDescriptorPool(VkApplication* app, light<spotLight>* object, uint32_t imageCount);
        void createLightDescriptorSets(VkApplication* app, light<spotLight>* object, uint32_t imageCount);

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

    void setApplication(VkApplication* app);
    void setImageProp(imageInfo* pInfo);
    void setEmptyTexture(texture* emptyTexture);
    void setCameraObject(camera* cameraObject);

    void createAttachments();
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createBaseDescriptorPool();
    void createBaseDescriptorSets();
    void createSkyboxDescriptorPool();
    void createSkyboxDescriptorSets();
    void createSecondDescriptorPool();
    void createSecondDescriptorSets();

    void updateSecondDescriptorSets();
    void updateLightDescriptorSets(light<spotLight>* object);

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t lightSourceCount, light<spotLight>** pplightSources);

    void updateUniformBuffer(uint32_t currentImage);
    void updateSkyboxUniformBuffer(uint32_t currentImage);
    void updateObjectUniformBuffer(uint32_t currentImage);

    void createStorageBuffers(VkApplication *app, uint32_t imageCount);
    void updateStorageBuffer(uint32_t currentImage, const glm::vec4& mousePosition);
    uint32_t readStorageBuffer(uint32_t currentImage);

    void bindBaseObject(object* newObject);
    void bindBloomObject(object* newObject);
    void bindOneColorObject(object* newObject);
    void bindStencilObject(object* newObject, float lineWidth, glm::vec4 lineColor);
    void bindSkyBoxObject(object* newObject, cubeTexture* texture);

    void removeBinds();

    void bindLightSource(light<spotLight>* object);

    void setStencilObject(object *oldObject);

    std::vector<attachments>&       getAttachments();
    std::vector<VkBuffer>&          getSceneBuffer();
    ShadowPassObjects               getObjects();
};

class postProcessing
{
private:
    VkApplication                       *app;
    uint32_t                            imageCount;
    VkSampleCountFlagBits               msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    uint32_t                            swapChainAttachmentCount = 1;
    VkSwapchainKHR                      swapChain;
    std::vector<attachments>            swapChainAttachments;
    VkFormat                            swapChainImageFormat;
    VkExtent2D                          swapChainExtent;

    uint32_t                            AttachmentCount = 1;
    std::vector<attachments>            Attachments;

    std::vector<attachments>            blitAttachments;
    attachments                         blitAttachment;

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
        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);    //Формат поверхности
        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);     //Режим презентации
        VkExtent2D chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities);                              //Экстент - это разрешение изображений
    void createImageViews();
    void createColorAttachments();
public:
    postProcessing();
    void setApplication(VkApplication *app);
    void destroy();

    void createAttachments(GLFWwindow* window, SwapChainSupportDetails swapChainSupport);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
        void createDescriptorSetLayout();
        void createFirstGraphicsPipeline();
        void createSecondGraphicsPipeline();

    void createDescriptorPool();
    void createDescriptorSets(std::vector<attachments>& Attachments, std::vector<VkBuffer>& uniformBuffers);

    void render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i);

    std::vector<attachments>        & getBlitAttachments();
    attachments                     & getBlitAttachment();
    VkSwapchainKHR                  & SwapChain();
    VkFormat                        & SwapChainImageFormat();
    VkExtent2D                      & SwapChainImageExtent();
    uint32_t                        & SwapChainImageCount();
};

#endif // GRAPHICS_H
