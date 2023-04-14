#ifndef TEXTURE_H
#define TEXTURE_H

#include <vulkan.h>

#include <string>
#include <vector>

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

struct iamge
{
    VkImage textureImage{VK_NULL_HANDLE};
    VkImageView textureImageView{VK_NULL_HANDLE};
    VkSampler textureSampler{VK_NULL_HANDLE};
    VkDeviceMemory textureImageMemory{VK_NULL_HANDLE};

    VkFormat format{VK_FORMAT_R8G8B8A8_UNORM};

    struct buffer{
        VkBuffer instance{VK_NULL_HANDLE};
        VkDeviceMemory memory{VK_NULL_HANDLE};
    }stagingBuffer;

    void destroy(VkDevice device);
    void destroyStagingBuffer(VkDevice device);
    void create(
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

class texture
{
protected:
    std::vector<std::string> path;

    float mipLevel{0.0f};
    uint32_t mipLevels{1};

    iamge image;

public:
    texture() = default;
    texture(const std::string & path);
    ~texture() = default;
    void destroy(VkDevice device);
    void destroyStagingBuffer(VkDevice device);

    void createTextureImage(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer,
            tinygltf::Image&    gltfimage);
    void createTextureImage(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer);
    void createTextureImageView(VkDevice device);
    void createTextureSampler(VkDevice device, struct textureSampler TextureSampler);
    void setMipLevel(float mipLevel);
    void setTextureFormat(VkFormat format);

    VkImageView* getTextureImageView();
    VkSampler*   getTextureSampler();
};


class cubeTexture: public texture
{
public:
    cubeTexture() = default;
    cubeTexture(const std::vector<std::string> & path);
    ~cubeTexture() = default;

    void createTextureImage(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer);
    void createTextureImageView(VkDevice device);
};

#endif // TEXTURE_H
