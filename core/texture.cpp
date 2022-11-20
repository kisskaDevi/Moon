#include "texture.h"
#include "operations.h"

#define STB_IMAGE_IMPLEMENTATION
#include <libs/stb-master/stb_image.h>
#include "libs/tinygltf-master/tiny_gltf.h"

#include <iostream>

texture::texture(){}

texture::texture(const std::string & TEXTURE_PATH)
{
    this->TEXTURE_PATH = TEXTURE_PATH;
}

texture::~texture()
{

}

void texture::destroy(VkDevice* device)
{
    image.destroy(device);
}

void texture::iamge::create(
        VkPhysicalDevice*   physicalDevice,
        VkDevice*           device,
        VkQueue*            queue,
        VkCommandPool*      commandPool,
        uint32_t&           mipLevels,
        int                 texWidth,
        int                 texHeight,
        VkDeviceSize        imageSize,
        void*               pixels)
{
    if(!pixels) throw std::runtime_error("failed to load texture image!");

    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    createBuffer(physicalDevice,device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
    void* data;
    vkMapMemory(*device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(*device, stagingBufferMemory);

    stbi_image_free(pixels);

    createImage(physicalDevice, device, texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT  | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

    transitionImageLayout(device,queue,commandPool, textureImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
    copyBufferToImage(device,queue,commandPool, stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

    vkDestroyBuffer(*device, stagingBuffer, nullptr);
    vkFreeMemory(*device, stagingBufferMemory, nullptr);

    generateMipmaps(physicalDevice,device,queue,commandPool, textureImage, format, texWidth, texHeight, mipLevels);
}

void texture::createTextureImage(
        VkPhysicalDevice*   physicalDevice,
        VkDevice*           device,
        VkQueue*            queue,
        VkCommandPool*      commandPool,
        tinygltf::Image&    gltfimage)
{
    stbi_uc* buffer = nullptr;
    VkDeviceSize bufferSize = 0;
    bool deleteBuffer = false;
    if (gltfimage.component == 3)
    {
        bufferSize = gltfimage.width * gltfimage.height * 4;
        buffer = new stbi_uc[bufferSize];
        stbi_uc* rgba = buffer;
        stbi_uc* rgb = &gltfimage.image[0];
        for (int32_t i = 0; i< gltfimage.width * gltfimage.height; ++i)
        {
            for (int32_t j = 0; j < 3; ++j) {
                rgba[j] = rgb[j];
            }
            rgba += 4;
            rgb += 3;
        }
        deleteBuffer = true;
    }
    else
    {
        buffer = &gltfimage.image[0];
        bufferSize = gltfimage.image.size();
    }

    if(!buffer)    throw std::runtime_error("failed to load texture image!");


    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(gltfimage.width, gltfimage.height)))) + 1;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    createBuffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
    void* data;
    vkMapMemory(*device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, buffer, static_cast<size_t>(bufferSize));
    vkUnmapMemory(*device, stagingBufferMemory);

    if (deleteBuffer){   delete[] buffer;}

    createImage(physicalDevice, device, gltfimage.width, gltfimage.height, mipLevels, VK_SAMPLE_COUNT_1_BIT, image.format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT  | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image.textureImage, image.textureImageMemory);

    transitionImageLayout(device,queue,commandPool, image.textureImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
    copyBufferToImage(device,queue,commandPool, stagingBuffer, image.textureImage, static_cast<uint32_t>(gltfimage.width), static_cast<uint32_t>(gltfimage.height));

    vkDestroyBuffer(*device, stagingBuffer, nullptr);
    vkFreeMemory(*device, stagingBufferMemory, nullptr);

    generateMipmaps(physicalDevice,device,queue,commandPool, image.textureImage, image.format, gltfimage.width, gltfimage.height, mipLevels);
}

void texture::createTextureImage(
        VkPhysicalDevice*   physicalDevice,
        VkDevice*           device,
        VkQueue*            queue,
        VkCommandPool*      commandPool)
{
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = 4 * texWidth * texHeight;

    if(!pixels)    throw std::runtime_error("failed to load texture image!");

    image.create(physicalDevice,device,queue,commandPool,mipLevels,texWidth,texHeight,imageSize,pixels);
}

void texture::createTextureImageView(VkDevice* device)
{
    image.textureImageView = createImageView(device, image.textureImage, image.format, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
}

void texture::createTextureSampler(VkDevice* device, struct textureSampler TextureSampler)
{
    VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = TextureSampler.magFilter;
        samplerInfo.minFilter = TextureSampler.minFilter;
        samplerInfo.addressModeU = TextureSampler.addressModeU;
        samplerInfo.addressModeV = TextureSampler.addressModeV;
        samplerInfo.addressModeW = TextureSampler.addressModeW;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.minLod = static_cast<float>(mipLevel*mipLevels);
        samplerInfo.maxLod = static_cast<float>(mipLevels);
        samplerInfo.mipLodBias = 0.0f;
    vkCreateSampler(*device, &samplerInfo, nullptr, &image.textureSampler);
}

void                texture::setMipLevel(float mipLevel){this->mipLevel = mipLevel;}
void                texture::setTextureFormat(VkFormat format){image.format = format;}

VkImageView*        texture::getTextureImageView(){return &image.textureImageView;}
VkSampler*          texture::getTextureSampler(){return &image.textureSampler;}

//cubeTexture

cubeTexture::cubeTexture(const std::vector<std::string> & TEXTURE_PATH)
{
    this->TEXTURE_PATH.resize(6);
    for(uint32_t i= 0;i<6;i++)
    {
        this->TEXTURE_PATH.at(i) = TEXTURE_PATH.at(i);
    }
}

cubeTexture::~cubeTexture(){}

void cubeTexture::destroy(VkDevice* device)
{
    image.destroy(device);
}

void cubeTexture::iamge::create(
        VkPhysicalDevice*   physicalDevice,
        VkDevice*           device,
        VkQueue*            queue,
        VkCommandPool*      commandPool,
        uint32_t&           mipLevels,
        int                 texWidth,
        int                 texHeight,
        VkDeviceSize        imageSize,
        void*               pixels[6])
{
    if(!pixels) throw std::runtime_error("failed to load texture image!");

    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    createBuffer(physicalDevice, device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    for(uint32_t i=0;i<6;i++)
    {
        void* data;
        vkMapMemory(*device, stagingBufferMemory, static_cast<uint32_t>(i*imageSize/6), static_cast<uint32_t>(imageSize/6), 0, &data);
            memcpy(data, pixels[i], static_cast<uint32_t>(imageSize/6));
        vkUnmapMemory(*device, stagingBufferMemory);
        stbi_image_free(pixels[i]);
    }

    createCubeImage(physicalDevice,device, texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT  | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

    transitionImageLayout(device,queue,commandPool, textureImage, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels,0);
    copyBufferToImage(device,queue,commandPool, stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight),0);

    vkDestroyBuffer(*device, stagingBuffer, nullptr);
    vkFreeMemory(*device, stagingBufferMemory, nullptr);

    generateMipmaps(physicalDevice,device,queue,commandPool, textureImage, format, texWidth, texHeight, mipLevels,0);
}

void cubeTexture::createTextureImage(
        VkPhysicalDevice*   physicalDevice,
        VkDevice*           device,
        VkQueue*            queue,
        VkCommandPool*      commandPool)
{
    int texWidth, texHeight, texChannels;
    void *pixels[6];
    VkDeviceSize imageSize = 0;
    for(uint32_t i=0;i<6;i++)
    {
        pixels[i]= stbi_load(TEXTURE_PATH.at(i).c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        imageSize += 4 * texWidth * texHeight;

        if(!pixels[i])  throw std::runtime_error("failed to load texture image!");
    }

    image.create(physicalDevice,device,queue,commandPool,mipLevels,texWidth,texHeight,imageSize,pixels);
}

void cubeTexture::createTextureImageView(VkDevice* device)
{
    image.textureImageView = createCubeImageView(device, image.textureImage, image.format, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
}

void cubeTexture::createTextureSampler(VkDevice* device, struct textureSampler TextureSampler)
{
    VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = TextureSampler.magFilter;
        samplerInfo.minFilter = TextureSampler.minFilter;
        samplerInfo.addressModeU = TextureSampler.addressModeU;
        samplerInfo.addressModeV = TextureSampler.addressModeV;
        samplerInfo.addressModeW = TextureSampler.addressModeW;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.minLod = static_cast<float>(mipLevel*mipLevels);
        samplerInfo.maxLod = static_cast<float>(mipLevels);
        samplerInfo.mipLodBias = 0.0f; // Optional
    vkCreateSampler(*device, &samplerInfo, nullptr, &image.textureSampler);
}

void                cubeTexture::setMipLevel(float mipLevel){this->mipLevel = mipLevel;}
void                cubeTexture::setTextureFormat(VkFormat format){image.format = format;}

VkImageView*        cubeTexture::getTextureImageView(){return &image.textureImageView;}
VkSampler*          cubeTexture::getTextureSampler(){return &image.textureSampler;}
