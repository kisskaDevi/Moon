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
    float mipLevel = 0.0f;
    uint32_t mipLevels;

    struct memory
    {
        VkDeviceMemory textureImageMemory;
        bool enable = false;
        void destroy(VkApplication* app)
        {
            if(enable){
                vkFreeMemory(app->getDevice(), textureImageMemory, nullptr);
                enable = false;
            }
        }
    }memory;

    struct iamge
    {
        VkImage textureImage;
        bool enable = false;
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        void destroy(VkApplication* app)
        {
            if(enable){
                vkDestroyImage(app->getDevice(), textureImage, nullptr);
                enable = false;
            }
        }
        void create(VkApplication* app, uint32_t& mipLevels, struct memory& memory, int texWidth, int texHeight, VkDeviceSize imageSize, void* pixels);
    }image;

    struct view
    {
        VkImageView textureImageView;
        bool enable = false;
        void destroy(VkApplication* app)
        {
            if(enable){
                vkDestroyImageView(app->getDevice(), textureImageView, nullptr);
                enable = false;
            }
        }
    }view;

    struct sampler
    {
        VkSampler textureSampler;
        bool enable = false;
        void destroy(VkApplication* app)
        {
            if(enable){
                vkDestroySampler(app->getDevice(),textureSampler,nullptr);
                enable = false;
            }
        }
    }sampler;

    std::string TEXTURE_PATH;
    VkApplication* app;
public:
    texture();
    texture(VkApplication* app);
    texture(VkApplication* app, const std::string & TEXTURE_PATH);
    ~texture();
    void destroy();

    void createTextureImage(tinygltf::Image& gltfimage);
    void createTextureImage();
    void createTextureImageView();
    void createTextureSampler(struct textureSampler TextureSampler);
    void setVkApplication(VkApplication* app);
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
        void destroy(VkApplication* app)
        {
            if(enable){
                vkFreeMemory(app->getDevice(), textureImageMemory, nullptr);
                enable = false;
            }
        }
    }memory;

    struct iamge
    {
        VkImage textureImage;
        bool enable = false;
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        void destroy(VkApplication* app)
        {
            if(enable){
                vkDestroyImage(app->getDevice(), textureImage, nullptr);
                enable = false;
            }
        }
        void create(VkApplication* app, uint32_t& mipLevels, struct memory& memory, int texWidth, int texHeight, VkDeviceSize imageSize, void* pixels[6]);
    }image;

    struct view
    {
        VkImageView textureImageView;
        bool enable = false;
        void destroy(VkApplication* app)
        {
            if(enable){
                vkDestroyImageView(app->getDevice(), textureImageView, nullptr);
                enable = false;
            }
        }
    }view;

    struct sampler
    {
        VkSampler textureSampler;
        bool enable = false;
        void destroy(VkApplication* app)
        {
            if(enable){
                vkDestroySampler(app->getDevice(),textureSampler,nullptr);
                enable = false;
            }
        }
    }sampler;

    std::vector<std::string> TEXTURE_PATH;
    VkApplication* app;

public:
    cubeTexture();
    cubeTexture(VkApplication* app);
    cubeTexture(VkApplication* app, const std::vector<std::string> & TEXTURE_PATH);
    ~cubeTexture();
    void destroy();

    void createTextureImage();
    void createTextureImageView();
    void createTextureSampler(struct textureSampler TextureSampler);
    void setVkApplication(VkApplication* app);
    void setMipLevel(float mipLevel);
    void setTextureFormat(VkFormat format);

    VkImageView & getTextureImageView();
    VkSampler   & getTextureSampler();
};

#endif // TEXTURE_H
