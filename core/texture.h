#ifndef TEXTURE_H
#define TEXTURE_H

#include <libs/vulkan/vulkan.h>

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

class texture
{
private:
    std::string TEXTURE_PATH;

    float mipLevel{0.0f};
    uint32_t mipLevels;

    struct iamge
    {
        VkImage textureImage{VK_NULL_HANDLE};
        VkImageView textureImageView{VK_NULL_HANDLE};
        VkSampler textureSampler{VK_NULL_HANDLE};
        VkDeviceMemory textureImageMemory{VK_NULL_HANDLE};

        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

        void destroy(VkDevice* device)
        {
            if(textureImage)        vkDestroyImage(*device, textureImage, nullptr);
            if(textureImageMemory)  vkFreeMemory(*device, textureImageMemory, nullptr);
            if(textureImageView)    vkDestroyImageView(*device, textureImageView, nullptr);
            if(textureSampler)      vkDestroySampler(*device,textureSampler,nullptr);
        }
        void create(
                VkPhysicalDevice*   physicalDevice,
                VkDevice*           device,
                VkQueue*            queue,
                VkCommandPool*      commandPool,
                uint32_t&           mipLevels,
                int                 texWidth,
                int                 texHeight,
                VkDeviceSize        imageSize,
                void*               pixels);
    }image;

public:
    texture();
    texture(const std::string & TEXTURE_PATH);
    ~texture();
    void destroy(VkDevice* device);

    void createTextureImage(
            VkPhysicalDevice*   physicalDevice,
            VkDevice*           device,
            VkQueue*            queue,
            VkCommandPool*      commandPool,
            tinygltf::Image&    gltfimage);
    void createTextureImage(
            VkPhysicalDevice*   physicalDevice,
            VkDevice*           device,
            VkQueue*            queue,
            VkCommandPool*      commandPool);
    void createTextureImageView(VkDevice* device);
    void createTextureSampler(VkDevice* device, struct textureSampler TextureSampler);
    void setMipLevel(float mipLevel);
    void setTextureFormat(VkFormat format);

    VkImageView* getTextureImageView();
    VkSampler*   getTextureSampler();
};


class cubeTexture
{
private:
    std::vector<std::string> TEXTURE_PATH;

    float mipLevel{0.0f};
    uint32_t mipLevels;

    struct iamge
    {
        VkImage textureImage{VK_NULL_HANDLE};
        VkImageView textureImageView{VK_NULL_HANDLE};
        VkSampler textureSampler{VK_NULL_HANDLE};
        VkDeviceMemory textureImageMemory{VK_NULL_HANDLE};

        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

        void destroy(VkDevice* device)
        {
            if(textureImage)        vkDestroyImage(*device, textureImage, nullptr);
            if(textureImageMemory)  vkFreeMemory(*device, textureImageMemory, nullptr);
            if(textureImageView)    vkDestroyImageView(*device, textureImageView, nullptr);
            if(textureSampler)      vkDestroySampler(*device,textureSampler,nullptr);
        }
        void create(
                VkPhysicalDevice*   physicalDevice,
                VkDevice*           device,
                VkQueue*            queue,
                VkCommandPool*      commandPool,
                uint32_t&           mipLevels,
                int                 texWidth,
                int                 texHeight,
                VkDeviceSize        imageSize,
                void*               pixels[6]);
    }image;

public:
    cubeTexture(const std::vector<std::string> & TEXTURE_PATH);
    ~cubeTexture();
    void destroy(VkDevice* device);

    void createTextureImage(
            VkPhysicalDevice*   physicalDevice,
            VkDevice*           device,
            VkQueue*            queue,
            VkCommandPool*      commandPool);
    void createTextureImageView(VkDevice* device);
    void createTextureSampler(VkDevice* device, struct textureSampler TextureSampler);
    void setMipLevel(float mipLevel);
    void setTextureFormat(VkFormat format);

    VkImageView* getTextureImageView();
    VkSampler*   getTextureSampler();
};

#endif // TEXTURE_H
