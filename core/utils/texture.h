#ifndef TEXTURE_H
#define TEXTURE_H

#include <vulkan.h>

#include "buffer.h"

#include <filesystem>

namespace moon::utils {

struct PhysicalDevice;

namespace tinygltf{
    struct Image;
};

struct TextureSampler {
    VkFilter magFilter;
    VkFilter minFilter;
    VkSamplerAddressMode addressModeU;
    VkSamplerAddressMode addressModeV;
    VkSamplerAddressMode addressModeW;
};

struct Iamge{
    VkImage textureImage{VK_NULL_HANDLE};
    VkImageView textureImageView{VK_NULL_HANDLE};
    VkSampler textureSampler{VK_NULL_HANDLE};
    VkDeviceMemory textureImageMemory{VK_NULL_HANDLE};

    VkFormat format{VK_FORMAT_R8G8B8A8_UNORM};

    Buffer stagingBuffer;

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
            void**              pixels,
            const uint32_t&     imageCount);
};

class Texture{
protected:
    std::vector<std::filesystem::path> path;

    float mipLevel{0.0f};
    uint32_t mipLevels{1};

    Iamge image;

public:
    Texture() = default;
    Texture(const std::filesystem::path & path);
    ~Texture() = default;
    void destroy(VkDevice device);
    void destroyStagingBuffer(VkDevice device);

    VkResult createTextureImage(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer,
            int                 width,
            int                 height,
            void**              buffer);
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
    VkResult createTextureSampler(VkDevice device, TextureSampler TextureSampler);
    void setMipLevel(float mipLevel);
    void setTextureFormat(VkFormat format);

    const VkImageView* getTextureImageView() const;
    const VkSampler*   getTextureSampler() const;
};


class CubeTexture: public Texture {
public:
    CubeTexture() = default;
    CubeTexture(const std::vector<std::filesystem::path> & path);
    ~CubeTexture() = default;

    void createTextureImage(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer);
    void createTextureImageView(VkDevice device);
};

Texture createEmptyTexture(const PhysicalDevice&, VkCommandPool, bool isBlack = true);

}
#endif // TEXTURE_H
