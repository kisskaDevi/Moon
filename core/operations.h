#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "vulkanCore.h"

//bufferOperations

uint32_t findMemoryType(
        VkPhysicalDevice                physicalDevice,
        uint32_t                        typeFilter,
        VkMemoryPropertyFlags           properties);

void createBuffer(
        VkApplication*                  app,
        VkDeviceSize                    size,
        VkBufferUsageFlags              usage,
        VkMemoryPropertyFlags           properties,
        VkBuffer&                       buffer,
        VkDeviceMemory&                 bufferMemory);

VkCommandBuffer beginSingleTimeCommands(
        VkApplication*                  app);

void endSingleTimeCommands(
        VkApplication*                  app,
        VkCommandBuffer                 commandBuffer);

void copyBuffer(
        VkApplication*                  app,
        VkBuffer                        srcBuffer,
        VkBuffer                        dstBuffer,
        VkDeviceSize                    size);

//textureOperations

void generateMipmaps(
        VkApplication*                  app,
        VkImage                         image,
        VkFormat                        imageFormat,
        int32_t                         texWidth,
        int32_t                         texHeight,
        uint32_t                        mipLevels);

void createImage(
        VkApplication*                  app,
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

void transitionImageLayout(
        VkApplication*                  app,
        VkImage                         image,
        VkFormat                        format,
        VkImageLayout                   oldLayout,
        VkImageLayout                   newLayout,
        uint32_t                        mipLevels);

void copyBufferToImage(
        VkApplication*                  app,
        VkBuffer                        buffer,
        VkImage                         image,
        uint32_t                        width,
        uint32_t                        height);

VkImageView createImageView(
        VkApplication*                  app,
        VkImage                         image,
        VkFormat                        format,
        VkImageAspectFlags              aspectFlags,
        uint32_t                        mipLevels);

//depthAttachmentsOperations

bool hasStencilComponent(
        VkFormat                        format);

VkFormat findDepthFormat(
        VkApplication*                  app);

VkFormat findSupportedFormat(
        VkApplication*                  app,
        const std::vector<VkFormat>&    candidates,
        VkImageTiling                   tiling,
        VkFormatFeatureFlags            features);

//shadersOperations

std::vector<char> readFile(
        const std::string&              filename);

VkShaderModule createShaderModule(
        VkApplication*                  app,
        const std::vector<char>&        code);

//deviceOperations

std::vector<QueueFamilyIndices> findQueueFamilies(
        VkPhysicalDevice                device,
        VkSurfaceKHR                    surface);

VkSampleCountFlagBits getMaxUsableSampleCount(
        VkPhysicalDevice                device);

void outDeviceInfo(
        std::vector<physicalDevice>&    physicalDevices);

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

#endif
