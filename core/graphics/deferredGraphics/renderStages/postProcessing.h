#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include<libs/vulkan/vulkan.h>
#include "../attachments.h"
#include "core/operations.h"

#include <string>

struct SwapChainSupportDetails;
struct QueueFamilyIndices;
class GLFWwindow;

class postProcessing
{
private:
    VkPhysicalDevice*                   physicalDevice;
    VkDevice*                           device;
    VkQueue*                            graphicsQueue;
    VkCommandPool*                      commandPool;
    QueueFamilyIndices                  queueFamilyIndices;
    VkSurfaceKHR*                       surface;

    imageInfo                           image;

    VkSwapchainKHR*                     swapChain;
    uint32_t                            swapChainAttachmentCount = 1;
    std::vector<attachments>            swapChainAttachments;

    uint32_t                            AttachmentCount = 1;
    std::vector<attachments>            Attachments;

    float                               blitFactor;
    uint32_t                            blitAttachmentCount;
    attachments*                        blitAttachments;
    attachments*                        blitAttachment;

    attachments*                        sslrAttachment;
    attachments*                        ssaoAttachment;

    VkRenderPass                        renderPass;
    std::vector<VkFramebuffer>          framebuffers;

    uint32_t                            transparentLayersCount;

    struct First{
        std::string                         ExternalPath;

        VkPipelineLayout                    PipelineLayout;
        VkPipeline                          Pipeline;
        VkDescriptorSetLayout               DescriptorSetLayout;
        VkDescriptorPool                    DescriptorPool;
        std::vector<VkDescriptorSet>        DescriptorSets;
    }first;

    struct Second{
        std::string                         ExternalPath;

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

    void setExternalPath(const std::string& path);
    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool, uint32_t graphicsFamily, uint32_t presentFamily);
    void setImageProp(imageInfo* pInfo);

    void setSwapChain(VkSwapchainKHR* swapChain);
    void setBlitAttachments(uint32_t blitAttachmentCount, attachments* blitAttachments);
    void setBlitAttachment(attachments* blitAttachment);
    void setSSLRAttachment(attachments* sslrAttachment);
    void setSSAOAttachment(attachments* ssaoAttachment);
    void setTransparentLayersCount(uint32_t TransparentLayersCount);

    void  setBlitFactor(const float& blitFactor);

    void createAttachments(GLFWwindow* window, SwapChainSupportDetails swapChainSupport, VkSurfaceKHR* surface);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
        void createDescriptorSetLayout();
        void createFirstGraphicsPipeline();
        void createSecondGraphicsPipeline();

    void createDescriptorPool();
    void createDescriptorSets(DeferredAttachments Attachments, std::vector<DeferredAttachments> transparentLayers);

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);

    VkFormat                        & SwapChainImageFormat();
    uint32_t                        & SwapChainImageCount();
};

#endif // POSTPROCESSING_H
