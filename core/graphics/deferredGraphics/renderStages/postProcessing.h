#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include<libs/vulkan/vulkan.h>
#include "../attachments.h"
#include "core/operations.h"

#include <string>

struct SwapChainSupportDetails;
struct QueueFamilyIndices;
class GLFWwindow;

class postProcessingGraphics
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

    attachments*                        blurAttachment;
    attachments*                        blitAttachments;
    attachments*                        sslrAttachment;
    attachments*                        ssaoAttachment;

    VkRenderPass                        renderPass;
    std::vector<VkFramebuffer>          framebuffers;

    struct PostProcessing{
        std::string                         ExternalPath;
        float                               blitFactor;
        uint32_t                            blitAttachmentCount;
        uint32_t                            transparentLayersCount;

        VkPipelineLayout                    PipelineLayout;
        VkPipeline                          Pipeline;
        VkDescriptorSetLayout               DescriptorSetLayout;
        VkDescriptorPool                    DescriptorPool;
        std::vector<VkDescriptorSet>        DescriptorSets;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
    }postProcessing;

    void createSwapChain(GLFWwindow* window, SwapChainSupportDetails swapChainSupport);
    void createImageViews();
public:
    postProcessingGraphics();
    void destroy();

    void setExternalPath(const std::string& path);
    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool, uint32_t graphicsFamily, uint32_t presentFamily);
    void setImageProp(imageInfo* pInfo);

    void setSwapChain(VkSwapchainKHR* swapChain);
    void setBlurAttachment(attachments* blurAttachment);
    void setBlitAttachments(uint32_t blitAttachmentCount, attachments* blitAttachments, float blitFactor);
    void setSSLRAttachment(attachments* sslrAttachment);
    void setSSAOAttachment(attachments* ssaoAttachment);
    void setTransparentLayersCount(uint32_t TransparentLayersCount);

    void createAttachments(GLFWwindow* window, SwapChainSupportDetails swapChainSupport, VkSurfaceKHR* surface);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createDescriptorPool();
    void createDescriptorSets();
    void updateDescriptorSets(DeferredAttachments Attachments, std::vector<DeferredAttachments> transparentLayers);

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);

    VkFormat& SwapChainImageFormat();
    uint32_t& SwapChainImageCount();
};

#endif // POSTPROCESSING_H
