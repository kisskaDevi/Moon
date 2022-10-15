#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <libs/vulkan/vulkan.h>
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>

#include <vector>
#include <string>
#include <optional> // нужна для вызова std::optional<uint32_t>

struct SwapChainSupportDetails{
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
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

//bufferOperations

uint32_t findMemoryType(
        VkPhysicalDevice                physicalDevice,
        uint32_t                        typeFilter,
        VkMemoryPropertyFlags           properties);

void createBuffer(
        VkPhysicalDevice*               physicalDevice,
        VkDevice*                       device,
        VkDeviceSize                    size,
        VkBufferUsageFlags              usage,
        VkMemoryPropertyFlags           properties,
        VkBuffer&                       buffer,
        VkDeviceMemory&                 bufferMemory);

VkCommandBuffer beginSingleTimeCommands(
        VkDevice*                       device,
        VkCommandPool*                  commandPool);

void endSingleTimeCommands(
        VkDevice*                       device,
        VkQueue*                        queue,
        VkCommandPool*                  commandPool,
        VkCommandBuffer*                commandBuffer);

void copyBuffer(
        VkDevice*                       device,
        VkQueue*                        queue,
        VkCommandPool*                  commandPool,
        VkBuffer                        srcBuffer,
        VkBuffer                        dstBuffer,
        VkDeviceSize                    size);

//textureOperations

void createImage(
        VkPhysicalDevice*               physicalDevice,
        VkDevice*                       device,
        uint32_t                        width,
        uint32_t                        height,
        uint32_t                        mipLevels,
        VkSampleCountFlagBits           numSamples,
        VkFormat                        format,
        VkImageTiling                   tiling,
        VkImageUsageFlags               usage,
        VkMemoryPropertyFlags           properties,
        VkImage&                        image,
        VkDeviceMemory&                 imageMemory);

void createImage(
        VkPhysicalDevice*               physicalDevice,
        VkDevice*                       device,
        uint32_t                        width,
        uint32_t                        height,
        uint32_t                        mipLevels,
        VkSampleCountFlagBits           numSamples,
        VkFormat                        format,
        VkImageTiling                   tiling,
        VkImageUsageFlags               usage,
        VkMemoryPropertyFlags           properties,
        VkImageLayout                   layout,
        VkImage&                        image,
        VkDeviceMemory&                 imageMemory);

void generateMipmaps(
        VkPhysicalDevice*               physicalDevice,
        VkDevice*                       device,
        VkQueue*                        queue,
        VkCommandPool*                  commandPool,
        VkImage                         image,
        VkFormat                        imageFormat,
        int32_t                         texWidth,
        int32_t                         texHeight,
        uint32_t                        mipLevels);

void generateMipmaps(
        VkCommandBuffer*                commandBuffer,
        VkImage                         image,
        int32_t                         texWidth,
        int32_t                         texHeight,
        uint32_t                        mipLevels);

void transitionImageLayout(
        VkDevice*                       device,
        VkQueue*                        queue,
        VkCommandPool*                  commandPool,
        VkImage                         image,
        VkImageLayout                   oldLayout,
        VkImageLayout                   newLayout,
        uint32_t                        mipLevels);


void transitionImageLayout(
        VkCommandBuffer*                commandBuffer,
        VkImage                         image,
        VkImageLayout                   oldLayout,
        VkImageLayout                   newLayout,
        uint32_t                        mipLevels);

void blitDown(
        VkCommandBuffer*                commandBuffer,
        VkImage                         srcImage,
        VkImage                         dstImage,
        uint32_t                        width,
        uint32_t                        height,
        float                           blitFactor);

void blitUp(
        VkCommandBuffer*                commandBuffer,
        VkImage                         srcImage,
        VkImage                         dstImage,
        uint32_t                        width,
        uint32_t                        height,
        float                           blitFactor);

void copyBufferToImage(
        VkDevice*                       device,
        VkQueue*                        queue,
        VkCommandPool*                  commandPool,
        VkBuffer                        buffer,
        VkImage                         image,
        uint32_t                        width,
        uint32_t                        height);

VkImageView createImageView(
        VkDevice*                       device,
        VkImage                         image,
        VkFormat                        format,
        VkImageAspectFlags              aspectFlags,
        uint32_t                        mipLevels);

void createImageView(
        VkDevice*                       device,
        VkImage                         image,
        VkFormat                        format,
        VkImageAspectFlags              aspectFlags,
        uint32_t                        mipLevels,
        VkImageView*                    imageView);


//cubeTextureOperations

void createCubeImage(
        VkPhysicalDevice*               physicalDevice,
        VkDevice*                       device,
        uint32_t                        width,
        uint32_t                        height,
        uint32_t                        mipLevels,
        VkSampleCountFlagBits           numSamples,
        VkFormat                        format,
        VkImageTiling                   tiling,
        VkImageUsageFlags               usage,
        VkMemoryPropertyFlags           properties,
        VkImage&                        image,
        VkDeviceMemory&                 imageMemory);

VkImageView createCubeImageView(
        VkDevice*                       device,
        VkImage                         image,
        VkFormat                        format,
        VkImageAspectFlags              aspectFlags,
        uint32_t                        mipLevels);

void transitionImageLayout(
        VkDevice*                       device,
        VkQueue*                        queue,
        VkCommandPool*                  commandPool,
        VkImage                         image,
        VkFormat                        format,
        VkImageLayout                   oldLayout,
        VkImageLayout                   newLayout,
        uint32_t                        mipLevels,
        uint32_t                        baseArrayLayer);

void copyBufferToImage(
        VkDevice*                       device,
        VkQueue*                        queue,
        VkCommandPool*                  commandPool,
        VkBuffer                        buffer,
        VkImage                         image,
        uint32_t                        width,
        uint32_t                        height,
        uint32_t                        baseArrayLayer);

void generateMipmaps(
        VkPhysicalDevice*               physicalDevice,
        VkDevice*                       device,
        VkQueue*                        queue,
        VkCommandPool*                  commandPool,
        VkImage                         image,
        VkFormat                        imageFormat,
        int32_t                         texWidth,
        int32_t                         texHeight,
        uint32_t                        mipLevels,
        uint32_t                        baseArrayLayer);

//depthAttachmentsOperations

bool hasStencilComponent(
        VkFormat                        format);

VkFormat findDepthFormat(
        VkPhysicalDevice*               physicalDevice);

VkFormat findDepthStencilFormat(
        VkPhysicalDevice*               physicalDevice);

VkFormat findSupportedFormat(
        VkPhysicalDevice*               physicalDevice,
        const std::vector<VkFormat>&    candidates,
        VkImageTiling                   tiling,
        VkFormatFeatureFlags            features);

//shadersOperations

std::vector<char> readFile(
        const std::string&              filename);

VkShaderModule createShaderModule(
        VkDevice*                       device,
        const std::vector<char>&        code);

//deviceOperations

std::vector<QueueFamilyIndices> findQueueFamilies(
        VkPhysicalDevice                device,
        VkSurfaceKHR                    surface);

VkSampleCountFlagBits getMaxUsableSampleCount(
        VkPhysicalDevice                device);

bool isDeviceSuitable(
        VkPhysicalDevice                device,
        VkSurfaceKHR                    surface,
        const std::vector<const char*>& deviceExtensions);

bool checkDeviceExtensionSupport(
        VkPhysicalDevice                device,
        const std::vector<const char*>& deviceExtensions);

SwapChainSupportDetails querySwapChainSupport(
        VkPhysicalDevice                device,
        VkSurfaceKHR                    surface);

//imageProp

VkSurfaceFormatKHR chooseSwapSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR>&  availableFormats);

VkPresentModeKHR chooseSwapPresentMode(
        const std::vector<VkPresentModeKHR>&    availablePresentModes);

VkExtent2D chooseSwapExtent(
        GLFWwindow* window,
        const VkSurfaceCapabilitiesKHR&         capabilities);

//decriptorSetLayout

void createObjectDescriptorSetLayout(
        VkDevice*                       device,
        VkDescriptorSetLayout*          descriptorSetLayout);

void createNodeDescriptorSetLayout(
        VkDevice*                       device,
        VkDescriptorSetLayout*          descriptorSetLayout);

void createMaterialDescriptorSetLayout(
        VkDevice*                       device,
        VkDescriptorSetLayout*          descriptorSetLayout);

void createSpotLightDescriptorSetLayout(
    VkDevice*                       device,
    VkDescriptorSetLayout*          descriptorSetLayout);

#endif
