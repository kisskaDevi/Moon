#include "texture.h"
#include <iostream>
#include "operations.h"

#define STB_IMAGE_IMPLEMENTATION
#include <libs/stb-master/stb_image.h>
#include "libs/tinygltf-master/tiny_gltf.h"

texture::texture(){}

texture::texture(VkApplication* app) : app(app){}

texture::texture(VkApplication* app, const std::string & TEXTURE_PATH) : app(app)
{
    this->TEXTURE_PATH = TEXTURE_PATH;
}

texture::~texture()
{

}

void texture::destroy()
{
    view.destroy(app);
    image.destroy(app);
    memory.destroy(app);
    sampler.destroy(app);
}

void texture::iamge::create(VkApplication* app, uint32_t& mipLevels, struct memory& memory, int texWidth, int texHeight, VkDeviceSize imageSize, void* pixels)
{
    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    createBuffer(app, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
    void* data;
    vkMapMemory(app->getDevice(), stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(app->getDevice(), stagingBufferMemory);

    stbi_image_free(pixels);

    createImage(app, texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT  | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, memory.textureImageMemory);

    transitionImageLayout(app, textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
    copyBufferToImage(app, stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

    vkDestroyBuffer(app->getDevice(), stagingBuffer, nullptr);
    vkFreeMemory(app->getDevice(), stagingBufferMemory, nullptr);

    generateMipmaps(app, textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);

    enable = true;
    memory.enable = true;
}

void texture::createTextureImage(tinygltf::Image& gltfimage)
{
    stbi_uc* buffer = nullptr;
    VkDeviceSize bufferSize = 0;
    bool deleteBuffer = false;
    if (gltfimage.component == 3)
    {
        // Most devices don't support RGB only on Vulkan so convert if necessary
        // TODO: Check actual format support and transform only if required
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

    if (!buffer) {
        throw std::runtime_error("failed to load texture image!");
    }

    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(gltfimage.width, gltfimage.height)))) + 1;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    createBuffer(app, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
    void* data;
    vkMapMemory(app->getDevice(), stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, buffer, static_cast<size_t>(bufferSize));
    vkUnmapMemory(app->getDevice(), stagingBufferMemory);

    if (deleteBuffer){   delete[] buffer;}

    createImage(app, gltfimage.width, gltfimage.height, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT  | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image.textureImage, memory.textureImageMemory);

    transitionImageLayout(app, image.textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
    copyBufferToImage(app, stagingBuffer, image.textureImage, static_cast<uint32_t>(gltfimage.width), static_cast<uint32_t>(gltfimage.height));

    vkDestroyBuffer(app->getDevice(), stagingBuffer, nullptr);
    vkFreeMemory(app->getDevice(), stagingBufferMemory, nullptr);

    generateMipmaps(app, image.textureImage, VK_FORMAT_R8G8B8A8_SRGB, gltfimage.width, gltfimage.height, mipLevels);

    image.enable = true;
    memory.enable = true;
}

void texture::createTextureImage()
{
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;      //Пиксели располагаются построчно с 4 байтами на пиксель

    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

    image.create(app,mipLevels,memory,texWidth,texHeight,imageSize,pixels);
}

void texture::createTextureImageView()
{
    view.textureImageView = createImageView(app, image.textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
    view.enable = true;
}

void texture::createTextureSampler(struct textureSampler TextureSampler)
{
    /* Сэмплеры настраиваются через VkSamplerCreateInfoструктуру, которая определяет
     * все фильтры и преобразования, которые она должна применять.*/

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = TextureSampler.magFilter;                           //поля определяют как интерполировать тексели, которые увеличенные
    samplerInfo.minFilter = TextureSampler.minFilter;                           //или минимизированы
    samplerInfo.addressModeU = TextureSampler.addressModeU;               //Режим адресации
    samplerInfo.addressModeV = TextureSampler.addressModeV;               //Обратите внимание, что оси называются U, V и W вместо X, Y и Z. Это соглашение для координат пространства текстуры.
    samplerInfo.addressModeW = TextureSampler.addressModeW;               //Повторение текстуры при выходе за пределы размеров изображения.
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;                                   //Чтобы выяснить, какое значение мы можем использовать, нам нужно получить свойства физического устройства
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;         //В этом borderColor поле указывается, какой цвет возвращается при выборке за пределами изображения в режиме адресации с ограничением по границе.
    samplerInfo.unnormalizedCoordinates = VK_FALSE;                     //поле определяет , какая система координат вы хотите использовать для адреса текселей в изображении
    samplerInfo.compareEnable = VK_FALSE;                               //Если функция сравнения включена, то тексели сначала будут сравниваться со значением,
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;                       //и результат этого сравнения используется в операциях фильтрации
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = static_cast<float>(mipLevel*mipLevels);
    samplerInfo.maxLod = static_cast<float>(mipLevels);
    samplerInfo.mipLodBias = 0.0f; // Optional

    if (vkCreateSampler(app->getDevice(), &samplerInfo, nullptr, &sampler.textureSampler) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create texture sampler!");
    }
    sampler.enable = true;
}

void                texture::setVkApplication(VkApplication* app){this->app=app;}
void                texture::setTextureNumber(uint32_t number){this->number = number;}
void                texture::setMipLevel(float mipLevel){this->mipLevel = mipLevel;}

VkImageView         & texture::getTextureImageView(){return view.textureImageView;}
VkSampler           & texture::getTextureSampler(){return sampler.textureSampler;}
uint32_t            & texture::getTextureNumber(){return number;}

//cubeTexture

cubeTexture::cubeTexture(){}
cubeTexture::cubeTexture(VkApplication* app) : app(app){}

cubeTexture::cubeTexture(VkApplication* app, const std::vector<std::string> & TEXTURE_PATH) : app(app)
{
    this->TEXTURE_PATH.resize(6);
    for(uint32_t i= 0;i<6;i++)
    {
        this->TEXTURE_PATH.at(i) = TEXTURE_PATH.at(i);
    }
}

cubeTexture::~cubeTexture(){}

void cubeTexture::destroy()
{
    view.destroy(app);
    image.destroy(app);
    memory.destroy(app);
    sampler.destroy(app);
}

void cubeTexture::iamge::create(VkApplication* app, uint32_t& mipLevels, struct memory& memory, int texWidth, int texHeight, VkDeviceSize imageSize, void* pixels[6])
{
    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    createBuffer(app, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    for(uint32_t i=0;i<6;i++)
    {
        void* data;
        vkMapMemory(app->getDevice(), stagingBufferMemory, static_cast<uint32_t>(i*imageSize/6), static_cast<uint32_t>(imageSize/6), 0, &data);
            memcpy(data, pixels[i], static_cast<uint32_t>(imageSize/6));
        vkUnmapMemory(app->getDevice(), stagingBufferMemory);
        stbi_image_free(pixels[i]);
    }

    createCubeImage(app, texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT  | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, memory.textureImageMemory);

    transitionImageLayout(app, textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels,0);
    copyBufferToImage(app, stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight),0);

    vkDestroyBuffer(app->getDevice(), stagingBuffer, nullptr);
    vkFreeMemory(app->getDevice(), stagingBufferMemory, nullptr);

    generateMipmaps(app, textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels,0);

    enable = true;
    memory.enable = true;
}

void cubeTexture::createTextureImage()
{
    int texWidth, texHeight, texChannels;
    void *pixels[6];
    VkDeviceSize imageSize = 0;
    for(uint32_t i=0;i<6;i++)
    {
        pixels[i]= stbi_load(TEXTURE_PATH.at(i).c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        imageSize += 4 * texWidth * texHeight;

        if (!pixels[i]) {
            throw std::runtime_error("failed to load texture image!");
        }
    }

    image.create(app,mipLevels,memory,texWidth,texHeight,imageSize,pixels);
}

void cubeTexture::createTextureImageView()
{
    view.textureImageView = createCubeImageView(app, image.textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
    view.enable = true;
}

void cubeTexture::createTextureSampler(struct textureSampler TextureSampler)
{
    /* Сэмплеры настраиваются через VkSamplerCreateInfoструктуру, которая определяет
     * все фильтры и преобразования, которые она должна применять.*/

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = TextureSampler.magFilter;                           //поля определяют как интерполировать тексели, которые увеличенные
    samplerInfo.minFilter = TextureSampler.minFilter;                           //или минимизированы
    samplerInfo.addressModeU = TextureSampler.addressModeU;               //Режим адресации
    samplerInfo.addressModeV = TextureSampler.addressModeV;               //Обратите внимание, что оси называются U, V и W вместо X, Y и Z. Это соглашение для координат пространства текстуры.
    samplerInfo.addressModeW = TextureSampler.addressModeW;               //Повторение текстуры при выходе за пределы размеров изображения.
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;                                   //Чтобы выяснить, какое значение мы можем использовать, нам нужно получить свойства физического устройства
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;         //В этом borderColor поле указывается, какой цвет возвращается при выборке за пределами изображения в режиме адресации с ограничением по границе.
    samplerInfo.unnormalizedCoordinates = VK_FALSE;                     //поле определяет , какая система координат вы хотите использовать для адреса текселей в изображении
    samplerInfo.compareEnable = VK_FALSE;                               //Если функция сравнения включена, то тексели сначала будут сравниваться со значением,
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;                       //и результат этого сравнения используется в операциях фильтрации
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = static_cast<float>(mipLevel*mipLevels);
    samplerInfo.maxLod = static_cast<float>(mipLevels);
    samplerInfo.mipLodBias = 0.0f; // Optional

    if (vkCreateSampler(app->getDevice(), &samplerInfo, nullptr, &sampler.textureSampler) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create texture sampler!");
    }
    sampler.enable = true;
}

void                cubeTexture::setVkApplication(VkApplication* app){this->app=app;}
void                cubeTexture::setTextureNumber(uint32_t number){this->number = number;}
void                cubeTexture::setMipLevel(float mipLevel){this->mipLevel = mipLevel;}

VkImageView         & cubeTexture::getTextureImageView(){return view.textureImageView;}
VkSampler           & cubeTexture::getTextureSampler(){return sampler.textureSampler;}
uint32_t            & cubeTexture::getTextureNumber(){return number;}
