#ifndef GRAPHICS_H
#define GRAPHICS_H

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3native.h>

#include <libs/glm/glm/glm.hpp>
#include <libs/glm/glm/gtc/matrix_transform.hpp>

#include <optional>         // нужна для вызова std::optional<uint32_t>
#include "attachments.h"
class VkApplication;

class texture;
class cubeTexture;
class object;
class camera;
struct gltfModel;
template <typename type>
class light;
class spotLight;
class pointLight;
template <> class light<spotLight>;
template <> class light<pointLight>;
struct Node;

const int MAX_LIGHT_SOURCE_COUNT = 8;

struct attachment
{
    VkImage image;
    VkDeviceMemory imageMemory;
    VkImageView imageView;
    void deleteAttachment(VkDevice * device)
    {
        vkDestroyImage(*device, image, nullptr);
        vkFreeMemory(*device, imageMemory, nullptr);
        vkDestroyImageView(*device, imageView, nullptr);
    }
};

struct UniformBufferObject
{
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 eyePosition;
};

struct SkyboxUniformBufferObject
{
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 model;
};

struct SecondUniformBufferObject
{
    alignas(16) glm::vec4 eyePosition;
};

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct QueueFamilyIndices
{
    //std::optional это оболочка, которая не содержит значения, пока вы ей что-то не присвоите.
    //В любой момент вы можете запросить, содержит ли он значение или нет, вызвав его has_value()функцию-член.

    std::optional<uint32_t> graphicsFamily;     //графикческое семейство очередей
    std::optional<uint32_t> presentFamily;      //семейство очередей показа

    bool isComplete() //если оба значения не пусты, а были записаны, выводит true
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

class graphics
{
private:
    VkApplication                   *app;
    texture                         *emptyTexture;
    glm::vec3                       cameraPosition;

    VkSampleCountFlagBits           msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    uint32_t                        imageCount;
    VkFormat                        imageFormat;
    VkExtent2D                      extent;

    std::vector<attachment>         colorAttachments;
    attachment                      depthAttachment;
    std::vector<attachments>        Attachments;

    VkRenderPass                    renderPass;
    std::vector<VkFramebuffer>      framebuffers;

    struct graphicsInfo{
        uint32_t                        imageCount;
        VkExtent2D                      extent;
        VkSampleCountFlagBits           msaaSamples;
        VkRenderPass                    renderPass;
    };

    struct Base{
        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;
        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorSetLayout           uniformBufferSetLayout;
        VkDescriptorSetLayout           uniformBlockSetLayout;
        VkDescriptorSetLayout           materialSetLayout;
        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;
        std::vector<VkBuffer>           uniformBuffers;
        std::vector<VkDeviceMemory>     uniformBuffersMemory;

        void Destroy(VkApplication  *app);
        void createPipeline(VkApplication *app, graphicsInfo info);
        void createDescriptorSetLayout(VkApplication *app);
        void createUniformBuffers(VkApplication *app, uint32_t imageCount);
    }base;

    struct Extension{
        VkPipeline                      bloomPipeline;
        VkPipeline                      godRaysPipeline;
        VkPipelineLayout                bloomPipelineLayout;
        VkPipelineLayout                godRaysPipelineLayout;

        void DestroyBloom(VkApplication  *app);
        void DestroyGodRays(VkApplication  *app);
        void createBloomPipeline(VkApplication *app, Base *base, graphicsInfo info);
        void createGodRaysPipeline(VkApplication *app, Base *base, graphicsInfo info);
    }extension;

    struct Skybox
    {
        cubeTexture                     *texture;
        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;
        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;
        std::vector<VkBuffer>           uniformBuffers;
        std::vector<VkDeviceMemory>     uniformBuffersMemory;

        void Destroy(VkApplication  *app);
        void createPipeline(VkApplication *app, graphicsInfo info);
        void createDescriptorSetLayout(VkApplication *app);
        void createUniformBuffers(VkApplication *app, uint32_t imageCount);
    }skybox;

    struct Second{
        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;
        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;
        std::vector<VkBuffer>           emptyUniformBuffers;
        std::vector<VkDeviceMemory>     emptyUniformBuffersMemory;
        std::vector<VkBuffer>           uniformBuffers;
        std::vector<VkDeviceMemory>     uniformBuffersMemory;

        void Destroy(VkApplication  *app);
        void createPipeline(VkApplication *app, graphicsInfo info);
        void createDescriptorSetLayout(VkApplication *app);
        void createUniformBuffers(VkApplication *app, uint32_t imageCount);
    }second;


    void createColorAttachments();
    void createDepthAttachment();
    void createResolveAttachments();

    void oneSampleRenderPass();
    void multiSampleRenderPass();
    void oneSampleFrameBuffer();
    void multiSampleFrameBuffer();
    void renderNode(Node* node, VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, VkDescriptorSet& objectDescriptorSet, VkPipelineLayout& layout);
public:
    graphics();
    void destroy();

    void setApplication(VkApplication *app);
    void setEmptyTexture(texture *emptyTexture);
    void setMSAASamples(VkSampleCountFlagBits msaaSamples);
    void setImageProp(uint32_t imageCount, VkFormat format, VkExtent2D extent);
    void setSkyboxTexture(cubeTexture * tex);

    void createAttachments();
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createBaseDescriptorPool(const std::vector<object*> & object3D);
    void createBaseDescriptorSets(const std::vector<object*> & object3D);

    void createSkyboxDescriptorPool();
    void createSkyboxDescriptorSets();

    void createSecondDescriptorPool();
    void createSecondDescriptorSets(const std::vector<light<spotLight>*> & lightSource);

    void render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i, std::vector<object*> & object3D, object & sky);

    void updateUniformBuffer(uint32_t currentImage, camera *cam, object *skybox);

    VkPipelineLayout                & PipelineLayout();
    VkPipelineLayout                & BloomSpritePipelineLayout();
    VkPipelineLayout                & GodRaysPipelineLayout();
    VkPipelineLayout                & SkyBoxPipelineLayout();

    VkPipeline                      & PipeLine();
    VkPipeline                      & BloomSpriteGraphicsPipeline();
    VkPipeline                      & GodRaysPipeline();
    VkPipeline                      & SkyBoxPipeLine();

    std::vector<attachments>        & getAttachments();
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
    void createSwapChain(SwapChainSupportDetails swapChainSupport);
        //Формат поверхности
        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
        //Режим презентации
        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
        //Экстент подкачки - это разрешение изображений цепочки подкачки, и оно почти всегда точно равно разрешению окна, в которое мы рисуем, в пикселях
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    void createImageViews();
    void createColorAttachments();
public:
    postProcessing();
    void setApplication(VkApplication *app);
    void setMSAASamples(VkSampleCountFlagBits msaaSamples);
    void destroy();

    void createAttachments(SwapChainSupportDetails swapChainSupport);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
        void createDescriptorSetLayout();
        void createFirstGraphicsPipeline();
        void createSecondGraphicsPipeline();

    void createDescriptorPool();
    void createDescriptorSets(std::vector<attachments> & Attachments);

    void render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i);

    VkSwapchainKHR                  & SwapChain();
    VkFormat                        & SwapChainImageFormat();
    VkExtent2D                      & SwapChainExtent();
    uint32_t                        & ImageCount();
};

#endif // GRAPHICS_H
