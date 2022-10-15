#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include<libs/vulkan/vulkan.h>

#include <libs/glm/glm/glm.hpp>
#include <libs/glm/glm/gtc/matrix_transform.hpp>

#include <string>
#include <iostream>
#include "attachments.h"
#include "core/operations.h"

struct postProcessingPushConst
{
    alignas(4) float                blitFactor;
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

#endif // POSTPROCESSING_H
