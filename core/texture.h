#ifndef TEXTURE_H
#define TEXTURE_H

#include "vulkanCore.h"
#include "operations.h"

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

    float mipLevel = 0.0f;
    uint32_t mipLevels;

    struct memory
    {
        VkDeviceMemory textureImageMemory;
        bool enable = false;
        void destroy(VkDevice* device)
        {
            if(enable){
                vkFreeMemory(*device, textureImageMemory, nullptr);
                enable = false;
            }
        }
    }memory;

    struct iamge
    {
        VkImage textureImage;
        bool enable = false;
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        void destroy(VkDevice* device)
        {
            if(enable){
                vkDestroyImage(*device, textureImage, nullptr);
                enable = false;
            }
        }
        void create(
                VkPhysicalDevice*   physicalDevice,
                VkDevice*           device,
                VkQueue*            queue,
                VkCommandPool*      commandPool,
                uint32_t& mipLevels, struct memory& memory, int texWidth, int texHeight, VkDeviceSize imageSize, void* pixels);
    }image;

    struct view
    {
        VkImageView textureImageView;
        bool enable = false;
        void destroy(VkDevice* device)
        {
            if(enable){
                vkDestroyImageView(*device, textureImageView, nullptr);
                enable = false;
            }
        }
    }view;

    struct sampler
    {
        VkSampler textureSampler;
        bool enable = false;
        void destroy(VkDevice* device)
        {
            if(enable){
                vkDestroySampler(*device,textureSampler,nullptr);
                enable = false;
            }
        }
    }sampler;

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

    VkImageView & getTextureImageView();
    VkSampler   & getTextureSampler();
};


class cubeTexture
{
private:
    float mipLevel = 0.0f;
    uint32_t mipLevels;

    struct memory
    {
        VkDeviceMemory textureImageMemory;
        bool enable = false;
        void destroy(VkDevice* device)
        {
            if(enable){
                vkFreeMemory(*device, textureImageMemory, nullptr);
                enable = false;
            }
        }
    }memory;

    struct iamge
    {
        VkImage textureImage;
        bool enable = false;
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        void destroy(VkDevice* device)
        {
            if(enable){
                vkDestroyImage(*device, textureImage, nullptr);
                enable = false;
            }
        }
        void create(
                VkPhysicalDevice*   physicalDevice,
                VkDevice*           device,
                VkQueue*            queue,
                VkCommandPool*      commandPool,
                uint32_t& mipLevels, struct memory& memory, int texWidth, int texHeight, VkDeviceSize imageSize, void* pixels[6]);
    }image;

    struct view
    {
        VkImageView textureImageView;
        bool enable = false;
        void destroy(VkDevice* device)
        {
            if(enable){
                vkDestroyImageView(*device, textureImageView, nullptr);
                enable = false;
            }
        }
    }view;

    struct sampler
    {
        VkSampler textureSampler;
        bool enable = false;
        void destroy(VkDevice* device)
        {
            if(enable){
                vkDestroySampler(*device,textureSampler,nullptr);
                enable = false;
            }
        }
    }sampler;

    std::vector<std::string> TEXTURE_PATH;
    VkApplication* app;

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
    void setVkApplication(VkApplication* app);
    void setMipLevel(float mipLevel);
    void setTextureFormat(VkFormat format);

    VkImageView & getTextureImageView();
    VkSampler   & getTextureSampler();
};

#endif // TEXTURE_H
