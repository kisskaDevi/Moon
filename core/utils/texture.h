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

class texture
{
private:
    std::string TEXTURE_PATH;

    float mipLevel{0.0f};
    uint32_t mipLevels{1};

    struct iamge
    {
        VkImage textureImage{VK_NULL_HANDLE};
        VkImageView textureImageView{VK_NULL_HANDLE};
        VkSampler textureSampler{VK_NULL_HANDLE};
        VkDeviceMemory textureImageMemory{VK_NULL_HANDLE};

        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

        struct buffer{
            VkBuffer instance{VK_NULL_HANDLE};
            VkDeviceMemory memory{VK_NULL_HANDLE};
        }stagingBuffer;

        void destroy(VkDevice* device)
        {
            destroyStagingBuffer(device);
            if(textureImage)        {vkDestroyImage(*device, textureImage, nullptr); textureImage = VK_NULL_HANDLE;}
            if(textureImageMemory)  {vkFreeMemory(*device, textureImageMemory, nullptr); textureImageMemory = VK_NULL_HANDLE;}
            if(textureImageView)    {vkDestroyImageView(*device, textureImageView, nullptr); textureImageView = VK_NULL_HANDLE;}
            if(textureSampler)      {vkDestroySampler(*device,textureSampler,nullptr); textureSampler = VK_NULL_HANDLE;}
        }
        void create(
                VkPhysicalDevice    physicalDevice,
                VkDevice            device,
                VkCommandBuffer     commandBuffer,
                uint32_t&           mipLevels,
                int                 texWidth,
                int                 texHeight,
                VkDeviceSize        imageSize,
                void*               pixels);
        void destroyStagingBuffer(VkDevice* device){
            if(stagingBuffer.instance) {vkDestroyBuffer(*device, stagingBuffer.instance, nullptr); stagingBuffer.instance = VK_NULL_HANDLE;}
            if(stagingBuffer.memory)   {vkFreeMemory(*device, stagingBuffer.memory, nullptr); stagingBuffer.memory = VK_NULL_HANDLE;}
        }
    }image;

public:
    texture();
    texture(const std::string & TEXTURE_PATH);
    ~texture();
    void destroy(VkDevice* device);
    void destroyStagingBuffer(VkDevice* device){
        image.destroyStagingBuffer(device);
    }

    void createTextureImage(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer,
            tinygltf::Image&    gltfimage);
    void createTextureImage(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer);
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
    uint32_t mipLevels{1};

    struct iamge
    {
        VkImage textureImage{VK_NULL_HANDLE};
        VkImageView textureImageView{VK_NULL_HANDLE};
        VkSampler textureSampler{VK_NULL_HANDLE};
        VkDeviceMemory textureImageMemory{VK_NULL_HANDLE};

        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

        struct buffer{
            VkBuffer instance{VK_NULL_HANDLE};
            VkDeviceMemory memory{VK_NULL_HANDLE};
        }stagingBuffer;

        void destroy(VkDevice* device)
        {
            destroyStagingBuffer(device);
            if(textureImage)        {vkDestroyImage(*device, textureImage, nullptr); textureImage = VK_NULL_HANDLE;}
            if(textureImageMemory)  {vkFreeMemory(*device, textureImageMemory, nullptr); textureImageMemory = VK_NULL_HANDLE;}
            if(textureImageView)    {vkDestroyImageView(*device, textureImageView, nullptr); textureImageView = VK_NULL_HANDLE;}
            if(textureSampler)      {vkDestroySampler(*device,textureSampler,nullptr); textureSampler = VK_NULL_HANDLE;}
        }
        void create(
                VkPhysicalDevice    physicalDevice,
                VkDevice            device,
                VkCommandBuffer     commandBuffer,
                uint32_t&           mipLevels,
                int                 texWidth,
                int                 texHeight,
                VkDeviceSize        imageSize,
                void*               pixels[6]);
        void destroyStagingBuffer(VkDevice* device){
            if(stagingBuffer.instance) {vkDestroyBuffer(*device, stagingBuffer.instance, nullptr); stagingBuffer.instance = VK_NULL_HANDLE;}
            if(stagingBuffer.memory)   {vkFreeMemory(*device, stagingBuffer.memory, nullptr); stagingBuffer.memory = VK_NULL_HANDLE;}
        }
    }image;

public:
    cubeTexture(const std::vector<std::string> & TEXTURE_PATH);
    ~cubeTexture();
    void destroy(VkDevice* device);
    void destroyStagingBuffer(VkDevice* device){
        image.destroyStagingBuffer(device);
    }

    void createTextureImage(
            VkPhysicalDevice    physicalDevice,
            VkDevice            device,
            VkCommandBuffer     commandBuffer);
    void createTextureImageView(VkDevice* device);
    void createTextureSampler(VkDevice* device, struct textureSampler TextureSampler);
    void setMipLevel(float mipLevel);
    void setTextureFormat(VkFormat format);

    VkImageView* getTextureImageView();
    VkSampler*   getTextureSampler();
};

#endif // TEXTURE_H
