#include "graphics.h"
#include "core/transformational/object.h"
#include "core/transformational/camera.h"
#include "core/transformational/light.h"
#include "core/graphics/shadowGraphics.h"
#include "core/texture.h"

#include <array>

deferredGraphics::deferredGraphics(){
    bloom.base = &base;
    oneColor.base = &base;
    stencil.base = &base;
}

std::vector<VkBuffer>&          deferredGraphics::getSceneBuffer()  { return base.sceneUniformBuffers;}
DeferredAttachments             deferredGraphics::getDeferredAttachments()
{
    DeferredAttachments deferredAttachments{};
        deferredAttachments.image            = &Attachments[0];
        deferredAttachments.blur             = &Attachments[1];
        deferredAttachments.bloom            = &Attachments[2];
        deferredAttachments.GBuffer.position = &Attachments[3];
        deferredAttachments.GBuffer.normal   = &Attachments[4];
        deferredAttachments.GBuffer.color    = &Attachments[5];
        deferredAttachments.GBuffer.emission = &Attachments[6];
        deferredAttachments.depth            = &depthAttachment;
    return deferredAttachments;
}

void                            deferredGraphics::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool)
{
    this->physicalDevice = physicalDevice;
    this->device = device;
    this->graphicsQueue = graphicsQueue;
    this->commandPool = commandPool;
}
void                            deferredGraphics::setEmptyTexture(std::string ZERO_TEXTURE){
    this->emptyTexture = new texture(ZERO_TEXTURE);
    emptyTexture->createTextureImage(physicalDevice,device,graphicsQueue,commandPool);
    emptyTexture->createTextureImageView(device);
    emptyTexture->createTextureSampler(device,{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
}
void                            deferredGraphics::setCameraObject(camera* cameraObject)                 { this->cameraObject = cameraObject;}
void                            deferredGraphics::setImageProp(imageInfo* pInfo)                        { this->image = *pInfo;}

void                            deferredGraphics::setMinAmbientFactor(const float& minAmbientFactor)    { spotLighting.minAmbientFactor = minAmbientFactor;}
void                            deferredGraphics::setScattering(const bool &enableScattering)           { spotLighting.enableScattering = enableScattering;}
void                            deferredGraphics::setTransparencyPass(const bool& transparencyPass)     { this->transparencyPass = transparencyPass;}

texture*                        deferredGraphics::getEmptyTexture()                                     { return emptyTexture;}

void deferredGraphics::destroyEmptyTexture(){
    emptyTexture->destroy(device);
    delete emptyTexture;
}

void deferredGraphics::setExternalPath(const std::string &path)
{
    base.ExternalPath = path;
    bloom.ExternalPath = path;
    oneColor.ExternalPath = path;
    stencil.ExternalPath = path;
    skybox.ExternalPath = path;
    spotLighting.ExternalPath = path;
}

void deferredGraphics::destroy()
{
    base.Destroy(device);
    bloom.Destroy(device);
    oneColor.Destroy(device);
    stencil.DestroyFirstPipeline(device);
    stencil.DestroySecondPipeline(device);
    skybox.Destroy(device);
    spotLighting.Destroy(device);

    for (size_t i = 0; i < storageBuffers.size(); i++)
    {
        vkDestroyBuffer(*device, storageBuffers[i], nullptr);
        vkFreeMemory(*device, storageBuffersMemory[i], nullptr);
    }

    vkDestroyRenderPass(*device, renderPass, nullptr);
    for(size_t i = 0; i< framebuffers.size();i++)
        vkDestroyFramebuffer(*device, framebuffers[i],nullptr);

    depthAttachment.deleteAttachment(device);
    depthAttachment.deleteSampler(device);
    for(size_t i=0;i<colorAttachments.size();i++)
        colorAttachments.at(i).deleteAttachment(device);
    for(size_t i=0;i<Attachments.size();i++)
        Attachments.at(i).deleteAttachment(device);
    for(size_t i=0;i<7;i++)
        Attachments.at(i).deleteSampler(device);
}

void deferredGraphics::createStorageBuffers(uint32_t imageCount)
{
    storageBuffers.resize(imageCount);
    storageBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
        createBuffer(   physicalDevice,
                        device,
                        sizeof(StorageBufferObject),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        storageBuffers[i],
                        storageBuffersMemory[i]);
}

void deferredGraphics::updateStorageBuffer(uint32_t currentImage, const glm::vec4& mousePosition)
{
    void* data;

    StorageBufferObject StorageUBO{};
        StorageUBO.mousePosition = mousePosition;
        StorageUBO.number = INT_FAST32_MAX;
        StorageUBO.depth = 1.0f;
    vkMapMemory(*device, storageBuffersMemory[currentImage], 0, sizeof(StorageUBO), 0, &data);
        memcpy(data, &StorageUBO, sizeof(StorageUBO));
    vkUnmapMemory(*device, storageBuffersMemory[currentImage]);
}

uint32_t deferredGraphics::readStorageBuffer(uint32_t currentImage)
{
    void* data;

    StorageBufferObject StorageUBO{};
    vkMapMemory(*device, storageBuffersMemory[currentImage], 0, sizeof(StorageUBO), 0, &data);
        memcpy(&StorageUBO, data, sizeof(StorageUBO));
    vkUnmapMemory(*device, storageBuffersMemory[currentImage]);

    return StorageUBO.number;
}

//=========================================================================//

void deferredGraphics::createAttachments()
{
    if(image.Samples!=VK_SAMPLE_COUNT_1_BIT)
        createColorAttachments();
    createDepthAttachment();
    createResolveAttachments();
}

void deferredGraphics::createColorAttachments()
{
    colorAttachments.resize(7);
    for(size_t i=0;i<3;i++)
    {
        createImage(        physicalDevice,
                            device,
                            image.Extent.width,
                            image.Extent.height,
                            1,
                            image.Samples,
                            image.Format,
                            VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            colorAttachments[i].image,
                            colorAttachments[i].imageMemory);

        createImageView(    device,
                            colorAttachments[i].image,
                            image.Format,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            1,
                            &colorAttachments.at(i).imageView);
    }
    for(size_t i=3;i<5;i++)
    {
        createImage(        physicalDevice,
                            device,
                            image.Extent.width,
                            image.Extent.height,
                            1,
                            image.Samples,
                            VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            colorAttachments[i].image,
                            colorAttachments[i].imageMemory);

        createImageView(    device,
                            colorAttachments[i].image,
                            VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            1,
                            &colorAttachments[i].imageView);
    }
    for(size_t i=5;i<colorAttachments.size();i++)
    {
        createImage(        physicalDevice,
                            device,
                            image.Extent.width,
                            image.Extent.height,
                            1,
                            image.Samples,
                            VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            colorAttachments[i].image,
                            colorAttachments[i].imageMemory);

        createImageView(    device,
                            colorAttachments[i].image,
                            VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            1,
                            &colorAttachments[i].imageView);
    }
}
void deferredGraphics::createDepthAttachment()
{
    createImage(        physicalDevice,
                        device,
                        image.Extent.width,
                        image.Extent.height,
                        1,
                        image.Samples,
                        findDepthStencilFormat(physicalDevice),
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        depthAttachment.image,
                        depthAttachment.imageMemory);

    createImageView(    device,
                        depthAttachment.image,
                        findDepthStencilFormat(physicalDevice),
                        VK_IMAGE_ASPECT_DEPTH_BIT,
                        1,
                        &depthAttachment.imageView);

    VkSamplerCreateInfo SamplerInfo{};
        SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        SamplerInfo.magFilter = VK_FILTER_LINEAR;                           //поля определяют как интерполировать тексели, которые увеличенные
        SamplerInfo.minFilter = VK_FILTER_LINEAR;                           //или минимизированы
        SamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Режим адресации
        SamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Обратите внимание, что оси называются U, V и W вместо X, Y и Z. Это соглашение для координат пространства текстуры.
        SamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Повторение текстуры при выходе за пределы размеров изображения.
        SamplerInfo.anisotropyEnable = VK_TRUE;
        SamplerInfo.maxAnisotropy = 1.0f;                                   //Чтобы выяснить, какое значение мы можем использовать, нам нужно получить свойства физического устройства
        SamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;         //В этом borderColor поле указывается, какой цвет возвращается при выборке за пределами изображения в режиме адресации с ограничением по границе.
        SamplerInfo.unnormalizedCoordinates = VK_FALSE;                     //поле определяет , какая система координат вы хотите использовать для адреса текселей в изображении
        SamplerInfo.compareEnable = VK_FALSE;                               //Если функция сравнения включена, то тексели сначала будут сравниваться со значением,
        SamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;                       //и результат этого сравнения используется в операциях фильтрации
        SamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        SamplerInfo.minLod = 0.0f;
        SamplerInfo.maxLod = 0.0f;
        SamplerInfo.mipLodBias = 0.0f;
    if (vkCreateSampler(*device, &SamplerInfo, nullptr, &depthAttachment.sampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics sampler!");
}
void deferredGraphics::createResolveAttachments()
{
    Attachments.resize(7);
    for(size_t i=0;i<2;i++)
    {
        Attachments[i].resize(image.Count);
        for(size_t Image=0; Image<image.Count; Image++)
        {
            createImage(        physicalDevice,
                                device,
                                image.Extent.width,
                                image.Extent.height,
                                1,
                                VK_SAMPLE_COUNT_1_BIT,
                                image.Format,
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                Attachments[i].image[Image],
                                Attachments[i].imageMemory[Image]);

            createImageView(    device,
                                Attachments[i].image[Image],
                                image.Format,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1,
                                &Attachments[i].imageView[Image]);
        }
    }
    Attachments[2].resize(image.Count);
    for(size_t Image=0; Image<image.Count; Image++)
    {
        createImage(        physicalDevice,
                            device,
                            image.Extent.width,
                            image.Extent.height,
                            1,
                            VK_SAMPLE_COUNT_1_BIT,
                            image.Format,
                            VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            Attachments[2].image[Image],
                            Attachments[2].imageMemory[Image]);

        createImageView(    device,
                            Attachments[2].image[Image],
                            image.Format,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            1,
                            &Attachments[2].imageView[Image]);
    }
    for(size_t i=3;i<5;i++)
    {
        Attachments[i].resize(image.Count);
        for(size_t Image=0; Image<image.Count; Image++)
        {
            createImage(        physicalDevice,
                                device,
                                image.Extent.width,
                                image.Extent.height,
                                1,
                                VK_SAMPLE_COUNT_1_BIT,
                                VK_FORMAT_R16G16B16A16_SFLOAT,
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                Attachments[i].image[Image],
                                Attachments[i].imageMemory[Image]);

            createImageView(    device,
                                Attachments[i].image[Image],
                                VK_FORMAT_R16G16B16A16_SFLOAT,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1,
                                &Attachments[i].imageView[Image]);
        }
    }
    for(size_t i=5;i<Attachments.size();i++)
    {
        Attachments[i].resize(image.Count);
        for(size_t Image=0; Image<image.Count; Image++)
        {
            createImage(        physicalDevice,
                                device,
                                image.Extent.width,
                                image.Extent.height,
                                1,
                                VK_SAMPLE_COUNT_1_BIT,
                                VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                Attachments[i].image[Image],
                                Attachments[i].imageMemory[Image]);

            createImageView(    device,
                                Attachments[i].image[Image],
                                VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1,
                                &Attachments[i].imageView[Image]);
        }
    }

    for(size_t i=0;i<7;i++)
    {
        VkSamplerCreateInfo SamplerInfo{};
            SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            SamplerInfo.magFilter = VK_FILTER_LINEAR;                           //поля определяют как интерполировать тексели, которые увеличенные
            SamplerInfo.minFilter = VK_FILTER_LINEAR;                           //или минимизированы
            SamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Режим адресации
            SamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Обратите внимание, что оси называются U, V и W вместо X, Y и Z. Это соглашение для координат пространства текстуры.
            SamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   //Повторение текстуры при выходе за пределы размеров изображения.
            SamplerInfo.anisotropyEnable = VK_TRUE;
            SamplerInfo.maxAnisotropy = 1.0f;                                   //Чтобы выяснить, какое значение мы можем использовать, нам нужно получить свойства физического устройства
            SamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;         //В этом borderColor поле указывается, какой цвет возвращается при выборке за пределами изображения в режиме адресации с ограничением по границе.
            SamplerInfo.unnormalizedCoordinates = VK_FALSE;                     //поле определяет , какая система координат вы хотите использовать для адреса текселей в изображении
            SamplerInfo.compareEnable = VK_FALSE;                               //Если функция сравнения включена, то тексели сначала будут сравниваться со значением,
            SamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;                       //и результат этого сравнения используется в операциях фильтрации
            SamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            SamplerInfo.minLod = 0.0f;
            SamplerInfo.maxLod = 0.0f;
            SamplerInfo.mipLodBias = 0.0f;
        if (vkCreateSampler(*device, &SamplerInfo, nullptr, &Attachments[i].sampler) != VK_SUCCESS)
            throw std::runtime_error("failed to create graphics sampler!");
    }
}

//=======================================RenderPass======================//

void deferredGraphics::createRenderPass()
{
    if(image.Samples==VK_SAMPLE_COUNT_1_BIT) oneSampleRenderPass();
    else                                     multiSampleRenderPass();
}
    void deferredGraphics::oneSampleRenderPass()
    {
        std::vector<VkAttachmentDescription> attachments;
        VkAttachmentDescription colorAttachment{};
            colorAttachment.format = image.Format;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments.push_back(colorAttachment);
            colorAttachment.format = image.Format;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments.push_back(colorAttachment);
            colorAttachment.format = image.Format;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        attachments.push_back(colorAttachment);
        for(size_t i=3;i<5;i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
                colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(colorAttachment);
        }
        for(size_t i=5;i<Attachments.size();i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = VK_FORMAT_R8G8B8A8_UNORM;
                colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(colorAttachment);
        }
        VkAttachmentDescription depthAttachment{};
            depthAttachment.format = findDepthStencilFormat(physicalDevice);
            depthAttachment.samples = image.Samples;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments.push_back(depthAttachment);

        //===========================first=====================================================//

        uint32_t index = 3;
        std::vector<VkAttachmentReference> firstAttachmentRef(4);
            for (size_t i=0;i<firstAttachmentRef.size();i++)
            {
                firstAttachmentRef.at(i).attachment = index;
                firstAttachmentRef.at(i).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                index++;
            }
        VkAttachmentReference firstDepthAttachmentRef{};
            firstDepthAttachmentRef.attachment = index++;
            firstDepthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        //===========================second====================================================//

        index = 0;
        std::vector<VkAttachmentReference> secondAttachmentRef(3);
            secondAttachmentRef.at(index).attachment = 0;
            secondAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        index++;
            secondAttachmentRef.at(index).attachment = 1;
            secondAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        index++;
            secondAttachmentRef.at(index).attachment = 2;
            secondAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        index = 0;
        std::vector<VkAttachmentReference> secondInAttachmentRef(5);
            secondInAttachmentRef.at(index).attachment = 3;
            secondInAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef.at(index).attachment = 4;
            secondInAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef.at(index).attachment = 5;
            secondInAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef.at(index).attachment = 6;
            secondInAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef.at(index).attachment = 7;
            secondInAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        //===========================subpass & dependency=========================================//
        index = 0;
        std::vector<VkSubpassDescription> subpass(2);
            subpass.at(index).pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.at(index).colorAttachmentCount = static_cast<uint32_t>(firstAttachmentRef.size());
            subpass.at(index).pColorAttachments = firstAttachmentRef.data();
            subpass.at(index).pDepthStencilAttachment = &firstDepthAttachmentRef;
        index++;
            subpass.at(index).pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.at(index).colorAttachmentCount = static_cast<uint32_t>(secondAttachmentRef.size());
            subpass.at(index).pColorAttachments = secondAttachmentRef.data();
            subpass.at(index).inputAttachmentCount = static_cast<uint32_t>(secondInAttachmentRef.size());
            subpass.at(index).pInputAttachments = secondInAttachmentRef.data();
            subpass.at(index).pDepthStencilAttachment = nullptr;

        index = 0;
        std::vector<VkSubpassDependency> dependency(2);
            dependency.at(index).srcSubpass = VK_SUBPASS_EXTERNAL;                                                                              //ссылка из исходного прохода (создавшего данные)
            dependency.at(index).dstSubpass = 0;                                                                                                //в целевой подпроход (поглощающий данные)
            dependency.at(index).srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;                                                              //задаёт как стадии конвейера в исходном проходе создают данные
            dependency.at(index).srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;                                                                     //поля задают как каждый из исходных проходов обращается к данным
            dependency.at(index).dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dependency.at(index).dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        index++;
            dependency.at(index).srcSubpass = 0;                                                                                                //ссылка из исходного прохода
            dependency.at(index).dstSubpass = 1;                                                                                                //в целевой подпроход (поглощающий данные)
            dependency.at(index).srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;                                                  //задаёт как стадии конвейера в исходном проходе создают данные
            dependency.at(index).srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;                                                          //поля задают как каждый из исходных проходов обращается к данным
            dependency.at(index).dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.at(index).dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

        //===========================createRenderPass====================================================//

        VkRenderPassCreateInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());         //количество структур VkAtachmentDescription, определяющих подключения, связанные с этим проходом рендеринга
            renderPassInfo.pAttachments = attachments.data();                                   //Каждая структура определяет одно изображение, которое будет использовано как входное, выходное или входное и выходное одновремнно для оного или нескольких проходо в данном редеринге
            renderPassInfo.subpassCount = static_cast<uint32_t>(subpass.size());
            renderPassInfo.pSubpasses = subpass.data();
            renderPassInfo.dependencyCount = static_cast<uint32_t>(subpass.size());
            renderPassInfo.pDependencies = dependency.data();

        if (vkCreateRenderPass(*device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)    //создаём проход рендеринга
            throw std::runtime_error("failed to create graphics render pass!");
    }
    void deferredGraphics::multiSampleRenderPass()
    {
        std::vector<VkAttachmentDescription> attachments;
        for(size_t i=0;i<3;i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = image.Format;
                colorAttachment.samples = image.Samples;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachments.push_back(colorAttachment);
        }
        for(size_t i=3;i<5;i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
                colorAttachment.samples = image.Samples;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachments.push_back(colorAttachment);
        }
        for(size_t i=5;i<colorAttachments.size();i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = VK_FORMAT_R8G8B8A8_UNORM;
                colorAttachment.samples = image.Samples;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachments.push_back(colorAttachment);
        }

        VkAttachmentDescription depthAttachment{};
            depthAttachment.format = findDepthStencilFormat(physicalDevice);
            depthAttachment.samples = image.Samples;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments.push_back(depthAttachment);

        for(size_t i=0;i<2;i++)
        {
            VkAttachmentDescription colorAttachmentResolve{};
                colorAttachmentResolve.format = image.Format;
                colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(colorAttachmentResolve);
        }
        VkAttachmentDescription colorAttachmentResolve{};
            colorAttachmentResolve.format = image.Format;
            colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        attachments.push_back(colorAttachmentResolve);
        for(size_t i=3;i<5;i++)
        {
            VkAttachmentDescription colorAttachmentResolve{};
                colorAttachmentResolve.format = VK_FORMAT_R16G16B16A16_SFLOAT;
                colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(colorAttachmentResolve);
        }
        for(size_t i=5;i<Attachments.size();i++)
        {
            VkAttachmentDescription colorAttachmentResolve{};
                colorAttachmentResolve.format = VK_FORMAT_R8G8B8A8_UNORM;
                colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(colorAttachmentResolve);
        }

        //===========================first=====================================================//

        uint32_t index = 3;
        std::vector<VkAttachmentReference> firstAttachmentRef(4);
            for (size_t i=0;i<firstAttachmentRef.size();i++)
            {
                firstAttachmentRef.at(i).attachment = index;
                firstAttachmentRef.at(i).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                index++;
            }

        VkAttachmentReference firstDepthAttachmentRef{};
            firstDepthAttachmentRef.attachment = index++;
            firstDepthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        index = colorAttachments.size()+4;
        std::vector<VkAttachmentReference> firstResolveRef(4);
            for (size_t i=0;i<firstResolveRef.size();i++)
            {
                firstResolveRef.at(i).attachment = index;
                firstResolveRef.at(i).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                index++;
            }

        //===========================second====================================================//

        index = 0;
        std::vector<VkAttachmentReference> secondAttachmentRef(3);
            secondAttachmentRef.at(index).attachment = 0;
            secondAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        index++;
            secondAttachmentRef.at(index).attachment = 1;
            secondAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        index++;
            secondAttachmentRef.at(index).attachment = 2;
            secondAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        index = 0;
        std::vector<VkAttachmentReference> secondResolveRef(3);
            secondResolveRef.at(index).attachment = colorAttachments.size()+1;
            secondResolveRef.at(index).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        index++;
            secondResolveRef.at(index).attachment = colorAttachments.size()+2;
            secondResolveRef.at(index).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        index++;
            secondResolveRef.at(index).attachment = colorAttachments.size()+3;
            secondResolveRef.at(index).layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        index = 0;
        std::vector<VkAttachmentReference> secondInAttachmentRef(4);
            secondInAttachmentRef.at(index).attachment = colorAttachments.size()+4;
            secondInAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef.at(index).attachment = colorAttachments.size()+5;
            secondInAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef.at(index).attachment = colorAttachments.size()+6;
            secondInAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef.at(index).attachment = colorAttachments.size()+7;
            secondInAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        //===========================subpass & dependency=========================================//
        index = 0;
        std::vector<VkSubpassDescription> subpass(2);
            subpass.at(index).pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.at(index).colorAttachmentCount = static_cast<uint32_t>(firstAttachmentRef.size());
            subpass.at(index).pColorAttachments = firstAttachmentRef.data();
            subpass.at(index).pDepthStencilAttachment = &firstDepthAttachmentRef;
            subpass.at(index).pResolveAttachments = firstResolveRef.data();
        index++;
            subpass.at(index).pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.at(index).colorAttachmentCount = static_cast<uint32_t>(secondAttachmentRef.size());
            subpass.at(index).pColorAttachments = secondAttachmentRef.data();
            subpass.at(index).inputAttachmentCount = static_cast<uint32_t>(secondInAttachmentRef.size());
            subpass.at(index).pInputAttachments = secondInAttachmentRef.data();
            subpass.at(index).pDepthStencilAttachment = &firstDepthAttachmentRef;
            subpass.at(index).pResolveAttachments = secondResolveRef.data();

        index = 0;
        std::vector<VkSubpassDependency> dependency(2);
            dependency.at(index).srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.at(index).dstSubpass = 0;
            dependency.at(index).srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependency.at(index).srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            dependency.at(index).dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dependency.at(index).dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        index++;
            dependency.at(index).srcSubpass = 0;
            dependency.at(index).dstSubpass = 1;
            dependency.at(index).srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.at(index).srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependency.at(index).dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dependency.at(index).dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

        //===========================createRenderPass====================================================//

        VkRenderPassCreateInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            renderPassInfo.pAttachments = attachments.data();
            renderPassInfo.subpassCount = static_cast<uint32_t>(subpass.size());
            renderPassInfo.pSubpasses = subpass.data();
            renderPassInfo.dependencyCount = static_cast<uint32_t>(subpass.size());
            renderPassInfo.pDependencies = dependency.data();

        if (vkCreateRenderPass(*device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)    //создаём проход рендеринга
            throw std::runtime_error("failed to create multiSample graphics render pass!");
    }

//===================Framebuffers===================================

void deferredGraphics::createFramebuffers()
{
    if(image.Samples == VK_SAMPLE_COUNT_1_BIT){
        oneSampleFrameBuffer();
    }else{
        multiSampleFrameBuffer();
    }
}
    void deferredGraphics::oneSampleFrameBuffer()
    {
        /* Фреймбуфер (буфер кадра) - эо объект, представляющий набор изображений, в который
         * графические конвейеры будут осуществлять рендеринг. Они затрагивают посление несколько
         * стадий в кнвейере: тесты глубины и трафарета, смешивание цветов, логические операции,
         * мультисемплинг и т.п. Фреймбуфер создаётся, используя ссылку на проход рендеринга, и может быть
         * использован с любым проходом рендеринга, имеющим похожую структуру подключений.*/

        framebuffers.resize(image.Count);
        for (size_t Image = 0; Image < image.Count; Image++)
        {
            std::vector<VkImageView> attachments;
            for(size_t i=0;i<Attachments.size();i++)
                attachments.push_back(Attachments[i].imageView[Image]);
            attachments.push_back(depthAttachment.imageView);

            VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;                                                                        //дескриптор объекта прохода рендеринга
                framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());                                    //число изображений
                framebufferInfo.pAttachments = attachments.data();                                                              //набор изображений, которые должны быть привязаны к фреймбуферу, передаётся через массив дескрипторов объектов VkImageView
                framebufferInfo.width = image.Extent.width;                                                                           //ширина изображения
                framebufferInfo.height = image.Extent.height;                                                                         //высота изображения
                framebufferInfo.layers = 1;                                                                                     //число слоёв

            if (vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &framebuffers[Image]) != VK_SUCCESS)  //создание буфера кадров
                throw std::runtime_error("failed to create graphics framebuffer!");
        }
    }
    void deferredGraphics::multiSampleFrameBuffer()
    {
        framebuffers.resize(image.Count);
        for (size_t Image = 0; Image < image.Count; Image++)
        {
            std::vector<VkImageView> attachments;
            for(size_t i=0;i<colorAttachments.size();i++)
                attachments.push_back(colorAttachments[i].imageView);
            attachments.push_back(depthAttachment.imageView);
            for(size_t i=0;i<Attachments.size();i++)
                attachments.push_back(Attachments[i].imageView[Image]);


            VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;                                                                            //дескриптор объекта прохода рендеринга
                framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());                                        //число изображений
                framebufferInfo.pAttachments = attachments.data();                                                                  //набор изображений, которые должны быть привязаны к фреймбуферу, передаётся через массив дескрипторов объектов VkImageView
                framebufferInfo.width = image.Extent.width;                                                                      //ширина изображения
                framebufferInfo.height = image.Extent.height;                                                                    //высота изображения
                framebufferInfo.layers = 1;                                                                                         //число слоёв

            if (vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &framebuffers[Image]) != VK_SUCCESS)  //создание буфера кадров
                throw std::runtime_error("failed to create multiSample graphics framebuffer!");
        }
    }

void deferredGraphics::createPipelines()
{
    base.createDescriptorSetLayout(device);
    base.createPipeline(device,&image,&renderPass);
    base.createUniformBuffers(physicalDevice,device,image.Count);
    bloom.createPipeline(device,&image,&renderPass);
    oneColor.createPipeline(device,&image,&renderPass);
    stencil.createFirstPipeline(device,&image,&renderPass);
    stencil.createSecondPipeline(device,&image,&renderPass);
    skybox.createDescriptorSetLayout(device);
    skybox.createPipeline(device,&image,&renderPass);
    skybox.createUniformBuffers(physicalDevice,device,image.Count);
    spotLighting.createDescriptorSetLayout(device);
    spotLighting.createPipeline(device,&image,&renderPass);
    spotLighting.createUniformBuffers(physicalDevice,device,image.Count);
}

void deferredGraphics::render(uint32_t frameNumber, VkCommandBuffer commandBuffers)
{
    std::vector<VkClearValue> clearValues;
    if(image.Samples == VK_SAMPLE_COUNT_1_BIT){
        clearValues.resize(8);
        for(size_t i=0;i<7;i++)
            clearValues[i].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[7].depthStencil = {1.0f, 0};
    }else{
        clearValues.resize(15);
        for(size_t i=0;i<7;i++)
            clearValues[i].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[7].depthStencil = {1.0f, 0};
        for(size_t i=8;i<15;i++)
            clearValues[i].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    }

    VkRenderPassBeginInfo drawRenderPassInfo{};
        drawRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        drawRenderPassInfo.renderPass = renderPass;
        drawRenderPassInfo.framebuffer = framebuffers[frameNumber];
        drawRenderPassInfo.renderArea.offset = {0, 0};
        drawRenderPassInfo.renderArea.extent = image.Extent;
        drawRenderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        drawRenderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers, &drawRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        primitiveCount = 0;

        skybox.render(frameNumber,commandBuffers);
        base.render(frameNumber,commandBuffers, primitiveCount);
        bloom.render(frameNumber,commandBuffers,primitiveCount);
        oneColor.render(frameNumber,commandBuffers,primitiveCount);
        stencil.render(frameNumber,commandBuffers,primitiveCount);

    vkCmdNextSubpass(commandBuffers, VK_SUBPASS_CONTENTS_INLINE);

        spotLighting.render(frameNumber,commandBuffers);

    vkCmdEndRenderPass(commandBuffers);
}

void deferredGraphics::updateUniformBuffer(uint32_t currentImage)
{
    void* data;

    UniformBufferObject baseUBO{};
        baseUBO.view = cameraObject->getViewMatrix();
        baseUBO.proj = cameraObject->getProjMatrix();
        baseUBO.eyePosition = glm::vec4(cameraObject->getTranslate(), 1.0);
        baseUBO.enableTransparency = transparencyPass ? 1.0 : 0.0;
    vkMapMemory(*device, base.sceneUniformBuffersMemory[currentImage], 0, sizeof(baseUBO), 0, &data);
        memcpy(data, &baseUBO, sizeof(baseUBO));
    vkUnmapMemory(*device, base.sceneUniformBuffersMemory[currentImage]);

    vkMapMemory(*device, spotLighting.uniformBuffersMemory[currentImage], 0, sizeof(baseUBO), 0, &data);
        memcpy(data, &baseUBO, sizeof(baseUBO));
    vkUnmapMemory(*device, spotLighting.uniformBuffersMemory[currentImage]);
}

void deferredGraphics::updateSkyboxUniformBuffer(uint32_t currentImage)
{
    if(skybox.objects.size()!=0)
    {
        void* data;

        SkyboxUniformBufferObject skyboxUBO{};
            skyboxUBO.view = cameraObject->getViewMatrix();
            skyboxUBO.proj = cameraObject->getProjMatrix();
            skyboxUBO.model = glm::translate(glm::mat4x4(1.0f),cameraObject->getTranslate())*skybox.objects[0]->ModelMatrix();
        vkMapMemory(*device, this->skybox.uniformBuffersMemory[currentImage], 0, sizeof(skyboxUBO), 0, &data);
            memcpy(data, &skyboxUBO, sizeof(skyboxUBO));
        vkUnmapMemory(*device, this->skybox.uniformBuffersMemory[currentImage]);
    }
}

void deferredGraphics::updateObjectUniformBuffer(uint32_t currentImage)
{
    for(size_t i=0;i<base.objects.size();i++)
        base.objects.at(i)->updateUniformBuffer(device,currentImage);
    for(size_t i=0;i<bloom.objects.size();i++)
        bloom.objects.at(i)->updateUniformBuffer(device,currentImage);
    for(size_t i=0;i<oneColor.objects.size();i++)
        oneColor.objects.at(i)->updateUniformBuffer(device,currentImage);
    for(size_t i=0;i<stencil.objects.size();i++)
        stencil.objects.at(i)->updateUniformBuffer(device,currentImage);
}

void deferredGraphics::bindBaseObject(object *newObject)
{
    base.objects.push_back(newObject);
}

void deferredGraphics::bindBloomObject(object *newObject)
{
    bloom.objects.push_back(newObject);
}

void deferredGraphics::bindOneColorObject(object *newObject)
{
    oneColor.objects.push_back(newObject);
}

void deferredGraphics::bindStencilObject(object *newObject)
{
    stencil.objects.push_back(newObject);
}

void deferredGraphics::bindSkyBoxObject(object *newObject, const std::vector<std::string>& TEXTURE_PATH)
{
    skybox.texture = new cubeTexture(TEXTURE_PATH);
    skybox.texture->setMipLevel(0.0f);
    skybox.texture->createTextureImage(physicalDevice,device,graphicsQueue,commandPool);
    skybox.texture->createTextureImageView(device);
    skybox.texture->createTextureSampler(device,{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
    skybox.objects.push_back(newObject);
}

bool deferredGraphics::removeBaseObject(object* object)
{
    bool result = false;
    for(uint32_t index = 0; index<base.objects.size(); index++){
        if(object==base.objects[index]){
            base.objects.erase(base.objects.begin()+index);
            result = true;
        }
    }
    return result;
}

bool deferredGraphics::removeBloomObject(object* object)
{
    bool result = false;
    for(uint32_t index = 0; index<bloom.objects.size(); index++){
        if(object==bloom.objects[index]){
            bloom.objects.erase(bloom.objects.begin()+index);
            result = true;
        }
    }
    return result;
}

bool deferredGraphics::removeOneColorObject(object* object)
{
    bool result = false;
    for(uint32_t index = 0; index<oneColor.objects.size(); index++){
        if(object==oneColor.objects[index]){
            oneColor.objects.erase(oneColor.objects.begin()+index);
            result = true;
        }
    }
    return result;
}

bool deferredGraphics::removeStencilObject(object* object)
{
    bool result = false;
    for(uint32_t index = 0; index<stencil.objects.size(); index++){
        if(object==stencil.objects[index]){
            stencil.objects.erase(stencil.objects.begin()+index);
            result = true;
        }
    }
    return result;
}

bool deferredGraphics::removeSkyBoxObject(object* object)
{
    bool result = false;
    for(uint32_t index = 0; index<skybox.objects.size(); index++){
        if(object==skybox.objects[index]){
            skybox.texture->destroy(device);
            delete skybox.texture;
            skybox.objects.erase(skybox.objects.begin()+index);
            result = true;
        }
    }
    return result;
}

void deferredGraphics::removeBinds()
{
    for(auto object: base.objects){
        object->destroy(device);
        object->destroyUniformBuffers(device);
    }
    for(auto object: bloom.objects){
        object->destroy(device);
        object->destroyUniformBuffers(device);
    }
    for(auto object: oneColor.objects){
        object->destroy(device);
        object->destroyUniformBuffers(device);
    }
    for(auto object: stencil.objects){
        object->destroy(device);
        object->destroyUniformBuffers(device);
    }

    base.objects.clear();
    bloom.objects.clear();
    oneColor.objects.clear();
    stencil.objects.clear();
}

void deferredGraphics::addSpotLightSource(spotLight *lightSource)
{
    spotLighting.lightSources.push_back(lightSource);
}

void deferredGraphics::removeSpotLightSource(spotLight *lightSource)
{
    for(uint32_t index = 0; index<spotLighting.lightSources.size(); index++){
        if(lightSource==spotLighting.lightSources[index]){
            spotLighting.lightSources.erase(spotLighting.lightSources.begin()+index);
        }
    }
}
