#ifndef TEXTURE_H
#define TEXTURE_H

#include "vulkanCore.h"

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

    uint32_t number;

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
    void setTextureNumber(uint32_t number);
    void setMipLevel(float mipLevel);

    VkImageView & getTextureImageView();
    VkSampler   & getTextureSampler();
    uint32_t    & getTextureNumber();
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

    uint32_t number;

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
    void setTextureNumber(uint32_t number);
    void setMipLevel(float mipLevel);

    VkImageView & getTextureImageView();
    VkSampler   & getTextureSampler();
    uint32_t    & getTextureNumber();
};

#endif // TEXTURE_H
