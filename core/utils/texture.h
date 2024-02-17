#ifndef TEXTURE_H
#define TEXTURE_H

#include <vulkan.h>

#include "buffer.h"

#include <filesystem>
#include <vector>

struct physicalDevice;

namespace tinygltf{
    struct Image;
};

struct textureSampler {
    VkFilter magFilter;
    VkFilter minFilter;
    VkSamplerAddressMode addressModeU;
    VkSamplerAddressMode addressModeV;
    VkSamplerAddressMode addressModeW;
};

struct iamge{
    VkImage textureImage{VK_NULL_HANDLE};
    VkImageView textureImageView{VK_NULL_HANDLE};
    VkSampler textureSampler{VK_NULL_HANDLE};
    VkDeviceMemory textureImageMemory{VK_NULL_HANDLE};

    VkFormat format{VK_FORMAT_R8G8B8A8_UNORM};

    buffer stagingBuffer;

    void destroy(VkDevice device);
    void destroyStagingBuffer(VkDevice device);
    VkResult create(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer,
            VkImageCreateFlags  flags,
            uint32_t&           mipLevels,
            int                 texWidth,
            int                 texHeight,
            VkDeviceSize        imageSize,
            unsigned char**     pixels,
            const uint32_t&     imageCount);
};

class texture{
protected:
    std::vector<std::filesystem::path> path;

    float mipLevel{0.0f};
    uint32_t mipLevels{1};

    iamge image;

public:
    texture() = default;
    texture(const std::filesystem::path & path);
    ~texture() = default;
    void destroy(VkDevice device);
    void destroyStagingBuffer(VkDevice device);

    VkResult createTextureImage(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer,
            tinygltf::Image&    gltfimage);
    VkResult createTextureImage(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer);
    VkResult createEmptyTextureImage(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer,
            bool                isBlack = true);
    VkResult createTextureImageView(VkDevice device);
    VkResult createTextureSampler(VkDevice device, struct textureSampler TextureSampler);
    void setMipLevel(float mipLevel);
    void setTextureFormat(VkFormat format);

    VkImageView* getTextureImageView();
    VkSampler*   getTextureSampler();
};


class cubeTexture: public texture{
public:
    cubeTexture() = default;
    cubeTexture(const std::vector<std::filesystem::path> & path);
    ~cubeTexture() = default;

    void createTextureImage(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer);
    void createTextureImageView(VkDevice device);
};

texture* createEmptyTexture(const physicalDevice&, VkCommandPool, bool isBlack = true);

#endif // TEXTURE_H
