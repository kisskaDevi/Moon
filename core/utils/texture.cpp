#include "texture.h"
#include "operations.h"

#include <stb_image.h>
#include "tiny_gltf.h"

#include <iostream>
#include <cstring>

void iamge::destroy(VkDevice device){
    destroyStagingBuffer(device);
    Texture::destroy(device, textureImage, textureImageMemory);
    if(textureImageView)    {vkDestroyImageView(device, textureImageView, nullptr); textureImageView = VK_NULL_HANDLE;}
    if(textureSampler)      {vkDestroySampler(device,textureSampler,nullptr); textureSampler = VK_NULL_HANDLE;}
}

void iamge::destroyStagingBuffer(VkDevice device){
    stagingBuffer.destroy(device);
}

VkResult iamge::create(
        VkPhysicalDevice    physicalDevice,
        VkDevice            device,
        VkCommandBuffer     commandBuffer,
        VkImageCreateFlags  flags,
        uint32_t&           mipLevels,
        int                 texWidth,
        int                 texHeight,
        VkDeviceSize        imageSize,
        stbi_uc**           pixels,
        const uint32_t&     imageCount)
{
    VkResult result = VK_SUCCESS;

    if(!pixels) throw std::runtime_error("failed to load texture image!");

    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    Buffer::create(physicalDevice, device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer.instance, &stagingBuffer.memory);
    Memory::instance().nameMemory(stagingBuffer.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", iamge::create, stagingBuffer");

    for(uint32_t i = 0, singleImageSize = static_cast<uint32_t>(imageSize/imageCount); i< imageCount; i++)
    {
        result = vkMapMemory(device, stagingBuffer.memory, i * singleImageSize, singleImageSize, 0, &stagingBuffer.map);
        debug::checkResult(result, "VkDeviceMemory : vkMapMemory result = " + std::to_string(result));
            std::memcpy(stagingBuffer.map, pixels[i], singleImageSize);
        vkUnmapMemory(device, stagingBuffer.memory);
        stagingBuffer.map = nullptr;
    }

    result = Texture::create(   physicalDevice,
                                device,
                                flags,
                                {static_cast<uint32_t>(texWidth),static_cast<uint32_t>(texHeight),1},
                                imageCount,
                                mipLevels,
                                VK_SAMPLE_COUNT_1_BIT,
                                format,
                                VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | (mipLevels > 1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0),
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                &textureImage,
                                &textureImageMemory);
    debug::checkResult(result, "VkImage : Texture::create result = " + std::to_string(result));

    Memory::instance().nameMemory(textureImageMemory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", attachments::createDepth, textureImageMemory");

    Texture::transitionLayout(commandBuffer, textureImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels, 0, imageCount);
    Texture::copyFromBuffer(commandBuffer, stagingBuffer.instance, textureImage, {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight),1},imageCount);
    if(mipLevels == 1){
        Texture::transitionLayout(commandBuffer, textureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mipLevels, 0, imageCount);
    } else {
        Texture::generateMipmaps(physicalDevice, commandBuffer, textureImage, format, texWidth, texHeight, mipLevels, 0, imageCount);
    }
    return result;
}

texture::texture(const std::filesystem::path& path) : path({path})
{}

void texture::destroy(VkDevice device){
    image.destroy(device);
}

void texture::destroyStagingBuffer(VkDevice device){
    image.destroyStagingBuffer(device);
}

VkResult texture::createTextureImage(
        VkPhysicalDevice    physicalDevice,
        VkDevice            device,
        VkCommandBuffer     commandBuffer,
        tinygltf::Image&    gltfimage)
{
    VkResult result = VK_SUCCESS;
    VkDeviceSize bufferSize = gltfimage.width * gltfimage.height * 4;
    stbi_uc* buffer[1];
    if (stbi_uc* rgb = &gltfimage.image[0]; gltfimage.component == 3){
        buffer[0] = new stbi_uc[bufferSize];

        for (int32_t i = 0; i< gltfimage.width * gltfimage.height; ++i){
            for (int32_t j = 0; j < 3; ++j) {
                buffer[0][4*i + j] = rgb[3*i + j];
            }
        }
        result = image.create(physicalDevice, device, commandBuffer, 0, mipLevels, gltfimage.width, gltfimage.height, bufferSize, buffer, 1);
        debug::checkResult(result, "iamge : create result = " + std::to_string(result));
        delete[] buffer[0];
    }else{
        buffer[0] = &gltfimage.image[0];
        result = image.create(physicalDevice, device, commandBuffer, 0, mipLevels, gltfimage.width, gltfimage.height, static_cast<VkDeviceSize>(gltfimage.image.size()), buffer, 1);
        debug::checkResult(result, "iamge : create result = " + std::to_string(result));
    }
    return result;
}

VkResult texture::createTextureImage(
        VkPhysicalDevice    physicalDevice,
        VkDevice            device,
        VkCommandBuffer     commandBuffer)
{
    VkResult result = VK_SUCCESS;
    int texWidth = 0, texHeight = 0, texChannels = 0;
    stbi_uc* pixels[1] = {stbi_load(path[0].string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha)};
    VkDeviceSize imageSize = 4 * texWidth * texHeight;

    if(!pixels[0])    throw std::runtime_error("failed to load texture image!");

    result = image.create(physicalDevice,device, commandBuffer, 0, mipLevels, texWidth, texHeight, imageSize, pixels, 1);
    debug::checkResult(result, "iamge : create result = " + std::to_string(result));
    stbi_image_free(pixels[0]);
    return result;
}

VkResult texture::createEmptyTextureImage(
        VkPhysicalDevice    physicalDevice,
        VkDevice            device,
        VkCommandBuffer     commandBuffer,
        bool                isBlack)
{
    VkResult result = VK_SUCCESS;
    stbi_uc* pixels[1] = {new stbi_uc[4]};
    for(size_t i = 0; i < 3; i++){
        pixels[0][i] = (isBlack ? 0 : 255);
    }
    pixels[0][3] = 255;

    result = image.create(physicalDevice,device, commandBuffer, 0, mipLevels, 1, 1, 4, pixels, 1);
    debug::checkResult(result, "iamge : create result = " + std::to_string(result));
    delete[] pixels[0];
    return result;
}

VkResult texture::createTextureImageView(VkDevice device)
{
    return Texture::createView( device,
                                VK_IMAGE_VIEW_TYPE_2D,
                                image.format,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                mipLevels,
                                0,
                                1,
                                image.textureImage,
                                &image.textureImageView);
}

VkResult texture::createTextureSampler(VkDevice device, struct textureSampler TextureSampler)
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
    return vkCreateSampler(device, &samplerInfo, nullptr, &image.textureSampler);
}

void                texture::setMipLevel(float mipLevel){this->mipLevel = mipLevel;}
void                texture::setTextureFormat(VkFormat format){image.format = format;}

VkImageView*        texture::getTextureImageView(){return &image.textureImageView;}
VkSampler*          texture::getTextureSampler(){return &image.textureSampler;}

//cubeTexture

cubeTexture::cubeTexture(const std::vector<std::filesystem::path>& path)
{
    for(const auto& p: path){
        this->path.push_back(p);
    }
}

void cubeTexture::createTextureImage(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer)
{
    int texWidth = 0, texHeight = 0, texChannels = 0;
    stbi_uc *pixels[6];
    VkDeviceSize imageSize = 0;
    for(uint32_t i=0;i<6;i++)
    {
        pixels[i]= stbi_load(path[i].string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        imageSize += 4 * texWidth * texHeight;

        if(!pixels[i]){
            throw std::runtime_error("failed to load texture image!");
        }
    }

    image.create(physicalDevice, device, commandBuffer, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, mipLevels, texWidth, texHeight, imageSize, pixels, 6);
    for(uint32_t i=0;i<6;i++){
        stbi_image_free(pixels[i]);
    }
}

void cubeTexture::createTextureImageView(VkDevice device)
{
    Texture::createView(    device,
                            VK_IMAGE_VIEW_TYPE_CUBE,
                            image.format,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            mipLevels,
                            0,
                            6,
                            image.textureImage,
                            &image.textureImageView);
}
