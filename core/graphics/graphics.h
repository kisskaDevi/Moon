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

struct skybox
{
    cubeTexture                     *texture;
    VkDescriptorSetLayout           descriptorSetLayout;
    VkDescriptorPool                descriptorPool;
    std::vector<VkDescriptorSet>    descriptorSets;
    VkPipeline                      pipeline;
    VkPipelineLayout                pipelineLayout;
    std::vector<VkBuffer>           uniformBuffers;
    std::vector<VkDeviceMemory>     uniformBuffersMemory;
};

class graphics
{
private:
    VkApplication                   *app;

    texture                         *emptyTexture;
    skybox                          skyBox;
    std::vector<VkBuffer>           emptyUniformBuffers;
    std::vector<VkDeviceMemory>     emptyUniformBuffersMemory;

    VkSampleCountFlagBits           msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    uint32_t                        imageCount;
    VkFormat                        swapChainImageFormat;
    VkExtent2D                      swapChainExtent;

    std::vector<attachment>         colorAttachments;
    attachment                      depthAttachment;
    std::vector<attachments>        Attachments;

    std::vector<VkBuffer>           uniformBuffers;
    std::vector<VkDeviceMemory>     uniformBuffersMemory;

    VkRenderPass                    renderPass;

    VkPipelineLayout                pipelineLayout;
    VkPipelineLayout                bloomSpritePipelineLayout;
    VkPipelineLayout                godRaysPipelineLayout;

    VkPipeline                      graphicsPipeline;
    VkPipeline                      bloomSpriteGraphicsPipeline;
    VkPipeline                      godRaysPipeline;

    std::vector<VkFramebuffer>      swapChainFramebuffers;

    VkDescriptorPool                descriptorPool;
    std::vector<VkDescriptorSet>    descriptorSets;

    VkDescriptorSetLayout           descriptorSetLayout;
    VkDescriptorSetLayout           uniformBufferSetLayout;
    VkDescriptorSetLayout           uniformBlockSetLayout;
    VkDescriptorSetLayout           materialSetLayout;

    glm::vec3                       cameraPosition;

    void oneSampleRenderPass();
    void multiSampleRenderPass();
    void oneSampleFrameBuffer();
    void multiSampleFrameBuffer();
    void renderNode(Node* node, VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, VkDescriptorSet& objectDescriptorSet, VkPipelineLayout& layout);
public:
    graphics();

    void setApplication(VkApplication *app);
    void setEmptyTexture(texture *emptyTexture);
    void setMSAASamples(VkSampleCountFlagBits msaaSamples);
    void setImageProp(uint32_t imageCount, VkFormat format, VkExtent2D extent);
    void setSkyboxTexture(cubeTexture * tex);

    void destroy();

    void createColorAttachments();
    void createDepthAttachment();
    void createAttachments();

    void createDrawRenderPass();

    void createDescriptorSetLayout();
    void createDescriptorPool(const std::vector<object*> & object3D);
    void createDescriptorSets(const std::vector<light<spotLight>*> & lightSource, const std::vector<object*> & object3D);


    void createSkyboxDescriptorSetLayout();
    void createSkyboxDescriptorPool();
    void createSkyboxDescriptorSets();

    void createGraphicsPipeline();
    void createBloomSpriteGraphicsPipeline();
    void createGodRaysGraphicsPipeline();
    void createSkyBoxPipeline();

    void createFramebuffers();

    void createUniformBuffers();
    void createSkyboxUniformBuffers();

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

    VkSwapchainKHR                  & SwapChain();
    VkFormat                        & SwapChainImageFormat();
    VkExtent2D                      & SwapChainExtent();
    uint32_t                        & ImageCount();

    std::vector<attachments>        & getAttachments();
    VkRenderPass                    & RenderPass();
    std::vector<VkFramebuffer>      & SwapChainFramebuffers();

    VkDescriptorSetLayout           & DescriptorSetLayout();
    VkDescriptorPool                & DescriptorPool();
    std::vector<VkDescriptorSet>    & DescriptorSets();

    std::vector<VkBuffer>           & UniformBuffers();
    std::vector<VkDeviceMemory>     & UniformBuffersMemory();
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

    VkPipelineLayout                    firstPipelineLayout;
    VkPipeline                          firstGraphicsPipeline;

    VkPipelineLayout                    secondPipelineLayout;
    VkPipeline                          secondGraphicsPipeline;

    VkDescriptorSetLayout               firstDescriptorSetLayout;
    VkDescriptorPool                    firstDescriptorPool;
    std::vector<VkDescriptorSet>        firstDescriptorSets;
    VkDescriptorSetLayout               secondDescriptorSetLayout;
    VkDescriptorPool                    secondDescriptorPool;
    std::vector<VkDescriptorSet>        secondDescriptorSets;

public:
    postProcessing();
    void setApplication(VkApplication *app);
    void setMSAASamples(VkSampleCountFlagBits msaaSamples);
    void destroy();

    //Создание цепочки обмена
    void createSwapChain();
        //Формат поверхности
        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
        //Режим презентации
        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
        //Экстент подкачки - это разрешение изображений цепочки подкачки, и оно почти всегда точно равно разрешению окна, в которое мы рисуем, в пикселях
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    void createImageViews();

    void createAttachments();

    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets(std::vector<attachments> & Attachments);

    void createRenderPass();
    void createFirstGraphicsPipeline();
    void createSecondGraphicsPipeline();
    void createFramebuffers();

    void render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i);

    VkSwapchainKHR                  & SwapChain();
    VkFormat                        & SwapChainImageFormat();
    VkExtent2D                      & SwapChainExtent();
    uint32_t                        & ImageCount();
};

#endif // GRAPHICS_H
