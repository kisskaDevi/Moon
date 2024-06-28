#include "texture.h"
#include "operations.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

#include <cmath>
#include <cstring>

namespace moon::utils {

void Image::swap(Image& other) noexcept {
    std::swap(image, other.image);
    std::swap(memory, other.memory);
    std::swap(imageView, other.imageView);
    std::swap(sampler, other.sampler);
    std::swap(format, other.format);
    std::swap(device, other.device);
    std::swap(width, other.width);
    std::swap(height, other.height);
    std::swap(channels, other.channels);
    std::swap(size, other.size);
    std::swap(cache, other.cache);
    std::swap(mipLevel, other.mipLevel);
    std::swap(mipLevels, other.mipLevels);
}

Image::Image(Image&& other) noexcept {
    swap(other);
}

Image& Image::operator=(Image&& other) noexcept {
    swap(other);
    return *this;
}

Image::~Image() {
    cache.destroy(device);
    texture::destroy(device, image, memory);
}

void Image::makeCache(
        VkPhysicalDevice            physicalDevice,
        VkDevice                    device,
        const std::vector<void*>&   buffers)
{
    this->device = device;
    if(width == -1 || height == -1 || channels == -1) throw std::runtime_error("[Image::makeCache] : texture sizes not init");

    buffer::create(physicalDevice, device, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &cache.instance, &cache.memory);
    Memory::instance().nameMemory(cache.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", Image::makeCache, cache");

    for (uint32_t i = 0, ds = 4 * width * height; i < buffers.size(); i++) {
        CHECK(vkMapMemory(device, cache.memory, i * ds, ds, 0, &cache.map));
        std::memcpy(cache.map, buffers[i], ds);
        vkUnmapMemory(device, cache.memory);
        cache.map = nullptr;
    }
}

VkResult Image::create(
        VkPhysicalDevice    physicalDevice,
        VkDevice            device,
        VkCommandBuffer     commandBuffer,
        VkImageCreateFlags  flags,
        const uint32_t&     imageCount,
        const TextureSampler& textureSampler)
{
    this->device = device;

    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
    VkResult result = texture::create(  physicalDevice,
                                        device,
                                        flags,
                                        {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
                                        imageCount,
                                        mipLevels,
                                        VK_SAMPLE_COUNT_1_BIT,
                                        format,
                                        VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | (mipLevels > 1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0),
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                        &image,
                                        &memory);
    Memory::instance().nameMemory(memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", Image::create, textureImageMemory");

    texture::transitionLayout(commandBuffer, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels, 0, imageCount);
    texture::copy(commandBuffer, cache.instance, image, {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1}, imageCount);
    if(mipLevels == 1){
        texture::transitionLayout(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mipLevels, 0, imageCount);
    } else {
        texture::generateMipmaps(physicalDevice, commandBuffer, image, format, width, height, mipLevels, 0, imageCount);
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = textureSampler.magFilter;
    samplerInfo.minFilter = textureSampler.minFilter;
    samplerInfo.addressModeU = textureSampler.addressModeU;
    samplerInfo.addressModeV = textureSampler.addressModeV;
    samplerInfo.addressModeW = textureSampler.addressModeW;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = static_cast<float>(mipLevel * mipLevels);
    samplerInfo.maxLod = static_cast<float>(mipLevels);
    samplerInfo.mipLodBias = 0.0f;

    VkImageViewType type = flags == VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
    result = std::max(result, imageView.create(device, image, type, format, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels, 0, imageCount));
    result = std::max(result, sampler.create(device, samplerInfo));
    return result;
}

Texture::Texture(const std::filesystem::path& path) : paths({path}) {}
Texture::Texture(const std::vector<std::filesystem::path>& path) : paths(path) {}

void Texture::swap(Texture& other) noexcept {
    std::swap(paths, other.paths);
    std::swap(image, other.image);
}

Texture::Texture(Texture&& other) noexcept {
    swap(other);
}

Texture& Texture::operator=(Texture&& other) noexcept {
    swap(other);
    return *this;
}

void Texture::destroyCache(){
    image.cache.destroy(image.device);
}

VkResult Texture::create(
    VkPhysicalDevice    physicalDevice,
    VkDevice            device,
    VkCommandBuffer     commandBuffer,
    int                 width,
    int                 height,
    void*               buffer,
    const TextureSampler& textureSampler)
{
    image.width = width;
    image.height = height;
    image.channels = 4;
    image.size = 4 * image.width * image.height;
    image.makeCache(physicalDevice, device, { buffer });
    return image.create(physicalDevice, device, commandBuffer, 0, 1, textureSampler);
}

VkResult Texture::create(
        VkPhysicalDevice    physicalDevice,
        VkDevice            device,
        VkCommandBuffer     commandBuffer,
        const TextureSampler& textureSampler)
{
    if(paths.empty()) throw std::runtime_error("[Texture::create] : no paths to texture");

    stbi_uc* buffer = stbi_load(paths.front().string().c_str(), &image.width, &image.height, &image.channels, STBI_rgb_alpha);
    image.size = 4 * image.width * image.height;
    if(!buffer) throw std::runtime_error("[Texture::create] : failed to load texture image!");
    image.makeCache(physicalDevice, device, { buffer });
    stbi_image_free(buffer);

    return image.create(physicalDevice, device, commandBuffer, 0, 1, textureSampler);
}

void Texture::setMipLevel(float mipLevel){image.mipLevel = mipLevel;}
void Texture::setTextureFormat(VkFormat format){image.format = format;}

const VkImageView Texture::imageView() const {return image.imageView;}
const VkSampler Texture::sampler() const {return image.sampler;}

CubeTexture::CubeTexture(const std::vector<std::filesystem::path>& path) : Texture(path) {}

VkResult CubeTexture::create(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, const TextureSampler& textureSampler)
{
    if (paths.size() != 6) throw std::runtime_error("[CubeTexture::create] : must be 6 images");

    int maxWidth = -1, maxHeight = -1, maxChannels = -1;
    std::vector<void*> buffers;
    for(uint32_t i = 0; i < 6; i++) {
        buffers.push_back(stbi_load(paths[i].string().c_str(), &image.width, &image.height, &image.channels, STBI_rgb_alpha));
        image.size += 4 * image.width * image.height;
        if (!buffers.back()) throw std::runtime_error("[CubeTexture::create] : failed to load texture image!");

        if (maxWidth == -1 && maxHeight == -1 && maxChannels == -1) {
            maxWidth = image.width; maxHeight = image.height; maxChannels = image.channels;
        }
        else if (maxWidth != image.width && maxHeight != image.height && maxChannels != image.channels) {
            throw std::runtime_error("[CubeTexture::create] : images must be same size!");
        }
    }

    image.makeCache(physicalDevice, device, buffers);
    for(auto& buffer : buffers) stbi_image_free(buffer);

    return image.create(physicalDevice, device, commandBuffer, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, 6, textureSampler);
}

Texture Texture::empty(const PhysicalDevice& device, VkCommandPool commandPool, bool isBlack){
    VkCommandBuffer commandBuffer = singleCommandBuffer::create(device.getLogical(),commandPool);
    Texture tex = Texture::empty(device, commandBuffer, isBlack);
    singleCommandBuffer::submit(device.getLogical(), device.getQueue(0, 0), commandPool, &commandBuffer);
    tex.destroyCache();
    return tex;
};

Texture Texture::empty(const PhysicalDevice& device, VkCommandBuffer commandBuffer, bool isBlack) {
    Texture tex;
    uint32_t buffer = isBlack ? 0xff000000 : 0xffffffff;
    int width = 1, height = 1;
    CHECK(tex.create(device.instance, device.getLogical(), commandBuffer, width, height, &buffer));
    return tex;
}

}
