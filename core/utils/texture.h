#ifndef TEXTURE_H
#define TEXTURE_H

#include <vulkan.h>
#include <filesystem>

#include "buffer.h"
#include "vkdefault.h"
#include "device.h"

namespace moon::utils {

struct TextureSampler {
    VkFilter magFilter{ VK_FILTER_LINEAR };
    VkFilter minFilter{ VK_FILTER_LINEAR };
    VkSamplerAddressMode addressModeU{ VK_SAMPLER_ADDRESS_MODE_REPEAT };
    VkSamplerAddressMode addressModeV{ VK_SAMPLER_ADDRESS_MODE_REPEAT };
    VkSamplerAddressMode addressModeW{ VK_SAMPLER_ADDRESS_MODE_REPEAT };
};

struct TextureImage {
    utils::vkDefault::Image image;
    utils::vkDefault::ImageView imageView;
    utils::vkDefault::Sampler sampler;

    VkDevice device{ VK_NULL_HANDLE };

    int width{ -1 };
    int height{ -1 };
    int channels{ -1 };
    VkDeviceSize size{ 0 };
    float mipLevel{ 0.0f };
    uint32_t mipLevels{ 1 };
    VkFormat format{ VK_FORMAT_R8G8B8A8_UNORM };

    Buffer cache;

    ~TextureImage();
    TextureImage() = default;
    TextureImage(const TextureImage&) = delete;
    TextureImage& operator=(const TextureImage&) = delete;
    TextureImage(TextureImage&&) noexcept;
    TextureImage& operator=(TextureImage&&) noexcept;
    void swap(TextureImage& other) noexcept;

    void makeCache(
            VkPhysicalDevice            physicalDevice,
            VkDevice                    device,
            const std::vector<void*>&   buffers);

    VkResult create(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer,
            VkImageCreateFlags  flags,
            const uint32_t&     imageCount,
            const TextureSampler& textureSampler);
};

class Texture{
protected:
    std::vector<std::filesystem::path> paths;
    TextureImage image;

    Texture(const std::vector<std::filesystem::path>& paths);

public:
    Texture(const std::filesystem::path& path);

    virtual ~Texture() = default;
    Texture() = default;
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&&) noexcept;
    Texture& operator=(Texture&&) noexcept;
    void swap(Texture& other) noexcept;

    void destroyCache();

    VkResult create(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer,
            int                 width,
            int                 height,
            void*               buffer,
            const TextureSampler& textureSampler = TextureSampler{});

    virtual VkResult create(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer,
            const TextureSampler& textureSampler = TextureSampler{});

    void setMipLevel(float mipLevel);
    void setTextureFormat(VkFormat format);

    const VkImageView imageView() const;
    const VkSampler sampler() const;

    static Texture empty(const PhysicalDevice&, VkCommandPool, bool isBlack = true);
    static Texture empty(const PhysicalDevice&, VkCommandBuffer, bool isBlack = true);
};

class CubeTexture: public Texture {
public:
    CubeTexture() = default;
    CubeTexture(const std::vector<std::filesystem::path>& path);
    ~CubeTexture() = default;

    VkResult create(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer,
            const TextureSampler& textureSampler = TextureSampler{}) override;
};

}
#endif // TEXTURE_H
