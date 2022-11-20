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
    VkPhysicalDevice*                   physicalDevice{nullptr};
    VkDevice*                           device{nullptr};
    VkQueue*                            graphicsQueue{nullptr};
    VkCommandPool*                      commandPool{nullptr};
    QueueFamilyIndices                  queueFamilyIndices;

    imageInfo                           image;

    uint32_t                            swapChainAttachmentCount{1};
    std::vector<attachments>            swapChainAttachments;

    attachments*                        blurAttachment{nullptr};
    attachments*                        blitAttachments{nullptr};
    attachments*                        sslrAttachment{nullptr};
    attachments*                        ssaoAttachment{nullptr};
    attachments*                        layersAttachment{nullptr};

    VkRenderPass                        renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>          framebuffers;

    struct PostProcessing{
        std::string                         ExternalPath;
        float                               blitFactor;
        uint32_t                            blitAttachmentCount;

        VkPipelineLayout                    PipelineLayout{VK_NULL_HANDLE};
        VkPipeline                          Pipeline{VK_NULL_HANDLE};
        VkDescriptorSetLayout               DescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorPool                    DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>        DescriptorSets;

        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
    }postProcessing;

public:
    postProcessingGraphics();
    void destroy();
    void destroySwapChainAttachments();

    void setExternalPath(const std::string& path);
    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool, uint32_t graphicsFamily, uint32_t presentFamily);
    void setImageProp(imageInfo* pInfo);

    void createSwapChain(VkSwapchainKHR* swapChain, GLFWwindow* window, SwapChainSupportDetails swapChainSupport, VkSurfaceKHR* surface);
    void createSwapChainAttachments(VkSwapchainKHR* swapChain);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createDescriptorPool();
    void createDescriptorSets();
    void updateDescriptorSets();

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);

    void setBlurAttachment(attachments* blurAttachment);
    void setBlitAttachments(uint32_t blitAttachmentCount, attachments* blitAttachments, float blitFactor);
    void setSSLRAttachment(attachments* sslrAttachment);
    void setSSAOAttachment(attachments* ssaoAttachment);
    void setLayersAttachment(attachments* layersAttachment);
};

#endif // POSTPROCESSING_H
