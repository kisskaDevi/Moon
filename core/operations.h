#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <libs/vulkan/vulkan.h>

#include <vector>
#include <string>
#include <optional>

class GLFWwindow;

struct QueueFamilyIndices
{
    std::optional<uint32_t>         graphicsFamily;
    std::optional<uint32_t>         presentFamily;
    bool isComplete(){return graphicsFamily.has_value() && presentFamily.has_value();}
};

namespace PhysicalDevice {

    void printMemoryProperties(
            VkPhysicalDeviceMemoryProperties memoryProperties);

    uint32_t findMemoryTypeIndex(
            VkPhysicalDevice                physicalDevice,
            uint32_t                        memoryTypeBits,
            VkMemoryPropertyFlags           properties);

    std::vector<QueueFamilyIndices> findQueueFamilies(
            VkPhysicalDevice                device,
            VkSurfaceKHR                    surface);

    VkSampleCountFlagBits queryingSampleCount(
            VkPhysicalDevice                device);

    bool isSuitable(
            VkPhysicalDevice                device,
            VkSurfaceKHR                    surface,
            const std::vector<const char*>& deviceExtensions);

    bool isExtensionsSupport(
            VkPhysicalDevice                device,
            const std::vector<const char*>& deviceExtensions);
}

namespace Buffer{

    VkResult create(
            VkPhysicalDevice                physicalDevice,
            VkDevice                        device,
            VkDeviceSize                    size,
            VkBufferUsageFlags              usage,
            VkMemoryPropertyFlags           properties,
            VkBuffer*                       buffer,
            VkDeviceMemory*                 bufferMemory);

    void copy(
            VkCommandBuffer                 commandBuffer,
            VkDeviceSize                    size,
            VkBuffer                        srcBuffer,
            VkBuffer                        dstBuffer);
}

namespace Texture {

    void transitionLayout(
            VkCommandBuffer                 commandBuffer,
            VkImage                         image,
            VkImageLayout                   oldLayout,
            VkImageLayout                   newLayout,
            uint32_t                        mipLevels,
            uint32_t                        baseArrayLayer,
            uint32_t                        arrayLayers);

    void copyFromBuffer(
            VkCommandBuffer                 commandBuffer,
            VkBuffer                        buffer,
            VkImage                         image,
            VkExtent3D                      extent,
            uint32_t                        layerCount);

    VkResult create(
            VkPhysicalDevice                physicalDevice,
            VkDevice                        device,
            VkImageCreateFlags              flags,
            VkExtent3D                      extent,
            uint32_t                        arrayLayers,
            uint32_t                        mipLevels,
            VkSampleCountFlagBits           numSamples,
            VkFormat                        format,
            VkImageLayout                   layout,
            VkImageUsageFlags               usage,
            VkMemoryPropertyFlags           properties,
            VkImage*                        image,
            VkDeviceMemory*                 imageMemory);

    VkResult createView(
            VkDevice                        device,
            VkImageViewType                 type,
            VkFormat                        format,
            VkImageAspectFlags              aspectFlags,
            uint32_t                        mipLevels,
            uint32_t                        baseArrayLayer,
            uint32_t                        layerCount,
            VkImage                         image,
            VkImageView*                    imageView);

    void generateMipmaps(
            VkPhysicalDevice                physicalDevice,
            VkCommandBuffer                 commandBuffer,
            VkImage                         image,
            VkFormat                        imageFormat,
            int32_t                         texWidth,
            int32_t                         texHeight,
            uint32_t                        mipLevels,
            uint32_t                        baseArrayLayer,
            uint32_t                        layerCount);

    void blitDown(
            VkCommandBuffer                 commandBuffer,
            VkImage                         srcImage,
            uint32_t                        srcMipLevel,
            VkImage                         dstImage,
            uint32_t                        dstMipLevel,
            uint32_t                        width,
            uint32_t                        height,
            uint32_t                        baseArrayLayer,
            uint32_t                        layerCount,
            float                           blitFactor);

    void blitUp(
            VkCommandBuffer                 commandBuffer,
            VkImage                         srcImage,
            uint32_t                        srcMipLevel,
            VkImage                         dstImage,
            uint32_t                        dstMipLevel,
            uint32_t                        width,
            uint32_t                        height,
            uint32_t                        baseArrayLayer,
            uint32_t                        layerCount,
            float                           blitFactor);
}

namespace SingleCommandBuffer {

    VkCommandBuffer create(
            VkDevice                        device,
            VkCommandPool                   commandPool);

    VkResult submit(
            VkDevice                        device,
            VkQueue                         queue,
            VkCommandPool                   commandPool,
            VkCommandBuffer*                commandBuffer);
}

namespace SwapChain {

    struct SupportDetails{
        VkSurfaceCapabilitiesKHR        capabilities{};
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR>   presentModes;
        bool isNotEmpty(){return !formats.empty() && !presentModes.empty();}
    };

    SupportDetails queryingSupport(
        VkPhysicalDevice                            device,
        VkSurfaceKHR                                surface);

    VkSurfaceFormatKHR queryingSurfaceFormat(
            const std::vector<VkSurfaceFormatKHR>&  availableFormats);

    VkPresentModeKHR queryingPresentMode(
            const std::vector<VkPresentModeKHR>&    availablePresentModes);

    VkExtent2D queryingExtent(
            GLFWwindow* window,
            const VkSurfaceCapabilitiesKHR&         capabilities);
}

namespace Image{

    VkFormat supportedFormat(
            VkPhysicalDevice                physicalDevice,
            const std::vector<VkFormat>&    candidates,
            VkImageTiling                   tiling,
            VkFormatFeatureFlags            features);

    VkFormat depthStencilFormat(
            VkPhysicalDevice                physicalDevice);
}

namespace ShaderModule {

    std::vector<char> readFile(
            const std::string&              filename);

    VkShaderModule create(
            VkDevice*                       device,
            const std::vector<char>&        code);
}

//  decriptorSetLayout

void createObjectDescriptorSetLayout(
        VkDevice*                       device,
        VkDescriptorSetLayout*          descriptorSetLayout);

void createSkyboxObjectDescriptorSetLayout(
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
