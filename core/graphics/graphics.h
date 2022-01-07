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
class object;
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
    VkPipeline                      skyBoxPipeline;
    VkPipeline                      bloomSpriteGraphicsPipeline;
    VkPipeline                      godRaysPipeline;

    std::vector<VkFramebuffer>      swapChainFramebuffers;

    VkDescriptorPool                descriptorPool;
    std::vector<VkDescriptorSet>    descriptorSets;

    VkDescriptorSetLayout           descriptorSetLayout;
    VkDescriptorSetLayout           uniformBufferSetLayout;
    VkDescriptorSetLayout           uniformBlockSetLayout;
    VkDescriptorSetLayout           materialSetLayout;

    void renderNode(Node* node, VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, VkDescriptorSet& objectDescriptorSet, VkPipelineLayout& layout);

public:
    graphics();
    void setApplication(VkApplication *app);
    void setEmptyTexture(texture *emptyTexture);
    void setMSAASamples(VkSampleCountFlagBits msaaSamples);
    void destroy();

    void setImageProp(uint32_t imageCount, VkFormat format, VkExtent2D extent);

    void createColorAttachments();
    void createDepthAttachment();
    void createAttachments();

    void createDrawRenderPass();

    void createDescriptorSetLayout();

    void createGraphicsPipeline();
    void createSkyBoxPipeline();
    void createBloomSpriteGraphicsPipeline();
    void createGodRaysGraphicsPipeline();

    void createFramebuffers();

    void createDescriptorPool(const std::vector<object*> & object3D);

    void createUniformBuffers();

    void createDescriptorSets(const std::vector<light<spotLight>*> & lightSource, const std::vector<object*> & object3D);

    void render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i, std::vector<object*> & object3D);

    VkPipelineLayout                & PipelineLayout();
    VkPipelineLayout                & BloomSpritePipelineLayout();
    VkPipelineLayout                & GodRaysPipelineLayout();

    VkPipeline                      & PipeLine();
    VkPipeline                      & SkyBoxPipeLine();
    VkPipeline                      & BloomSpriteGraphicsPipeline();
    VkPipeline                      & GodRaysPipeline();

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

    VkSwapchainKHR                      swapChain;
    std::vector<attachments>            swapChainAttachments;
    VkFormat                            swapChainImageFormat;
    VkExtent2D                          swapChainExtent;

    VkSampleCountFlagBits               msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    uint32_t                            imageCount;

    VkRenderPass                        renderPass;

    VkPipelineLayout                    pipelineLayout;
    VkPipeline                          graphicsPipeline;

    std::vector<VkFramebuffer>          framebuffers;


    VkDescriptorSetLayout               descriptorSetLayout;
    VkDescriptorPool                    descriptorPool;
    std::vector<VkDescriptorSet>        descriptorSets;

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

    void createDescriptorSetLayout();
    void createDescriptorPool(std::vector<attachments> & Attachments);
    void createDescriptorSets(std::vector<attachments> & Attachments);

    void createRenderPass();
    void createGraphicsPipeline();
    void createFramebuffers();

    void render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i);

    VkSwapchainKHR                  & SwapChain();
    VkFormat                        & SwapChainImageFormat();
    VkExtent2D                      & SwapChainExtent();
    uint32_t                        & ImageCount();

    std::vector<attachments>        & getAttachments();

    VkRenderPass                    & RenderPass();
    VkPipelineLayout                & PipelineLayout();
    VkPipeline                      & GraphicsPipeline();
    std::vector<VkFramebuffer>      & Framebuffers();

    VkDescriptorSetLayout           & DescriptorSetLayout();
    VkDescriptorPool                & DescriptorPool();
    std::vector<VkDescriptorSet>    & DescriptorSets();
};

#endif // GRAPHICS_H
