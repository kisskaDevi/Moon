#include "graphics.h"
#include "core/operations.h"
#include "core/transformational/object.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/camera.h"
#include "core/transformational/light.h"
#include "core/graphics/shadowGraphics.h"

#include <array>

graphics::graphics(){
    bloom.base = &base;
    oneColor.base = &base;
    stencil.base = &base;
}

std::vector<attachments>&       graphics::getAttachments()  { return Attachments;}
std::vector<VkBuffer>&          graphics::getSceneBuffer()  { return base.sceneUniformBuffers;}
ShadowPassObjects               graphics::getObjects()      { return ShadowPassObjects{&base.objects,&oneColor.objects,&stencil.objects};}

void                            graphics::setApplication(VkApplication* app)    { this->app = app;}
void                            graphics::setEmptyTexture(texture* emptyTexture){ this->emptyTexture = emptyTexture;}
void                            graphics::setCameraObject(camera* cameraObject) { this->cameraObject = cameraObject;}
void                            graphics::setImageProp(imageInfo* pInfo)        { this->image = *pInfo;}

void graphics::destroy()
{
    base.Destroy(app);
    bloom.Destroy(app);
    oneColor.Destroy(app);
    stencil.DestroyFirstPipeline(app);
    stencil.DestroySecondPipeline(app);
    skybox.Destroy(app);
    second.Destroy(app);

    for (size_t i = 0; i < storageBuffers.size(); i++)
    {
        vkDestroyBuffer(app->getDevice(), storageBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), storageBuffersMemory[i], nullptr);
    }

    vkDestroyRenderPass(app->getDevice(), renderPass, nullptr);
    for(size_t i = 0; i< framebuffers.size();i++)
        vkDestroyFramebuffer(app->getDevice(), framebuffers[i],nullptr);

    depthAttachment.deleteAttachment(&app->getDevice());
    for(size_t i=0;i<colorAttachments.size();i++)
        colorAttachments.at(i).deleteAttachment(&app->getDevice());
    for(size_t i=0;i<Attachments.size();i++)
        Attachments.at(i).deleteAttachment(&app->getDevice());
    for(size_t i=0;i<7;i++)
        Attachments.at(i).deleteSampler(&app->getDevice());

    depthAttachment.deleteSampler(&app->getDevice());
}

void graphics::createStorageBuffers(VkApplication *app, uint32_t imageCount)
{
    storageBuffers.resize(imageCount);
    storageBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
        createBuffer(app, sizeof(StorageBufferObject),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     storageBuffers[i], storageBuffersMemory[i]);
}

void graphics::updateStorageBuffer(uint32_t currentImage, const glm::vec4& mousePosition)
{
    void* data;

    StorageBufferObject StorageUBO{};
        StorageUBO.mousePosition = mousePosition;
        StorageUBO.number = INT_FAST32_MAX;
        StorageUBO.depth = 1.0f;
    vkMapMemory(app->getDevice(), storageBuffersMemory[currentImage], 0, sizeof(StorageUBO), 0, &data);
        memcpy(data, &StorageUBO, sizeof(StorageUBO));
    vkUnmapMemory(app->getDevice(), storageBuffersMemory[currentImage]);
}

uint32_t graphics::readStorageBuffer(uint32_t currentImage)
{
    void* data;

    StorageBufferObject StorageUBO{};
    vkMapMemory(app->getDevice(), storageBuffersMemory[currentImage], 0, sizeof(StorageUBO), 0, &data);
        memcpy(&StorageUBO, data, sizeof(StorageUBO));
    vkUnmapMemory(app->getDevice(), storageBuffersMemory[currentImage]);

    return StorageUBO.number;
}

//=========================================================================//

void graphics::createAttachments()
{
    if(image.Samples!=VK_SAMPLE_COUNT_1_BIT)
        createColorAttachments();
    createDepthAttachment();
    createResolveAttachments();
}

void graphics::createColorAttachments()
{
    colorAttachments.resize(7);
    for(size_t i=0;i<3;i++)
    {
        createImage(        app,
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

        createImageView(    app,
                            colorAttachments[i].image,
                            image.Format,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            1,
                            &colorAttachments.at(i).imageView);
    }
    for(size_t i=3;i<5;i++)
    {
        createImage(        app,
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

        createImageView(    app,
                            colorAttachments[i].image,
                            VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            1,
                            &colorAttachments[i].imageView);
    }
    for(size_t i=5;i<colorAttachments.size();i++)
    {
        createImage(        app,
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

        createImageView(    app,
                            colorAttachments[i].image,
                            VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            1,
                            &colorAttachments[i].imageView);
    }
}
void graphics::createDepthAttachment()
{
    createImage(        app,
                        image.Extent.width,
                        image.Extent.height,
                        1,
                        image.Samples,
                        findDepthStencilFormat(app),
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        depthAttachment.image,
                        depthAttachment.imageMemory);

    createImageView(    app,
                        depthAttachment.image,
                        findDepthStencilFormat(app),
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
    if (vkCreateSampler(app->getDevice(), &SamplerInfo, nullptr, &depthAttachment.sampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics sampler!");
}
void graphics::createResolveAttachments()
{
    Attachments.resize(7);
    for(size_t i=0;i<2;i++)
    {
        Attachments[i].resize(image.Count);
        for(size_t Image=0; Image<image.Count; Image++)
        {
            createImage(        app,
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

            createImageView(    app,
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
        createImage(        app,
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

        createImageView(    app, Attachments[2].image[Image],
                            image.Format, VK_IMAGE_ASPECT_COLOR_BIT,
                            1, &Attachments[2].imageView[Image]);
    }
    for(size_t i=3;i<5;i++)
    {
        Attachments[i].resize(image.Count);
        for(size_t Image=0; Image<image.Count; Image++)
        {
            createImage(        app,
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

            createImageView(    app,
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
            createImage(        app,
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

            createImageView(    app,
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
        if (vkCreateSampler(app->getDevice(), &SamplerInfo, nullptr, &Attachments[i].sampler) != VK_SUCCESS)
            throw std::runtime_error("failed to create graphics sampler!");
    }
}

//=======================================RenderPass======================//

void graphics::createRenderPass()
{
    if(image.Samples==VK_SAMPLE_COUNT_1_BIT)
    {
        oneSampleRenderPass();
    }else{
        multiSampleRenderPass();
    }
}
    void graphics::oneSampleRenderPass()
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
            depthAttachment.format = findDepthStencilFormat(app);
            depthAttachment.samples = image.Samples;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
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
        std::vector<VkAttachmentReference> secondInAttachmentRef(4);
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
            subpass.at(index).pDepthStencilAttachment = &firstDepthAttachmentRef;

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

        if (vkCreateRenderPass(app->getDevice(), &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)    //создаём проход рендеринга
            throw std::runtime_error("failed to create graphics render pass!");
    }
    void graphics::multiSampleRenderPass()
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
            depthAttachment.format = findDepthStencilFormat(app);
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
        std::vector<VkAttachmentReference> secondInAttachmentRef(5);
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

        if (vkCreateRenderPass(app->getDevice(), &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)    //создаём проход рендеринга
            throw std::runtime_error("failed to create multiSample graphics render pass!");
    }

//===================Framebuffers===================================

void graphics::createFramebuffers()
{
    if(image.Samples == VK_SAMPLE_COUNT_1_BIT){
        oneSampleFrameBuffer();
    }else{
        multiSampleFrameBuffer();
    }
}
    void graphics::oneSampleFrameBuffer()
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

            if (vkCreateFramebuffer(app->getDevice(), &framebufferInfo, nullptr, &framebuffers[Image]) != VK_SUCCESS)  //создание буфера кадров
                throw std::runtime_error("failed to create graphics framebuffer!");
        }
    }
    void graphics::multiSampleFrameBuffer()
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

            if (vkCreateFramebuffer(app->getDevice(), &framebufferInfo, nullptr, &framebuffers[Image]) != VK_SUCCESS)  //создание буфера кадров
                throw std::runtime_error("failed to create multiSample graphics framebuffer!");
        }
    }

void graphics::createPipelines()
{
    base.createDescriptorSetLayout(app);
    base.createPipeline(app,&image,&renderPass);
    base.createUniformBuffers(app,image.Count);
    bloom.createPipeline(app,&image,&renderPass);
    oneColor.createPipeline(app,&image,&renderPass);
    stencil.createFirstPipeline(app,&image,&renderPass);
    stencil.createSecondPipeline(app,&image,&renderPass);
    skybox.createDescriptorSetLayout(app);
    skybox.createPipeline(app,&image,&renderPass);
    skybox.createUniformBuffers(app,image.Count);
    second.createDescriptorSetLayout(app);
    second.createPipeline(app,&image,&renderPass);
    second.createUniformBuffers(app,image.Count);
}

void graphics::render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t lightSourceCount, light<spotLight>** pplightSources)
{
    std::array<VkClearValue, 8> clearValues{};
        for(size_t i=0;i<7;i++)
            clearValues[i].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[7].depthStencil = {1.0f, 0};

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

        second.render(frameNumber,commandBuffers,lightSourceCount,pplightSources);

    vkCmdEndRenderPass(commandBuffers);
}

void graphics::updateUniformBuffer(uint32_t currentImage)
{
    void* data;

    UniformBufferObject baseUBO{};
        baseUBO.view = cameraObject->getViewMatrix();
        baseUBO.proj = glm::perspective(glm::radians(45.0f), (float) image.Extent.width / (float) image.Extent.height, 0.1f, 1000.0f);
        baseUBO.proj[1][1] *= -1;
        baseUBO.eyePosition = glm::vec4(cameraObject->getTranslate(), 1.0);
    vkMapMemory(app->getDevice(), base.sceneUniformBuffersMemory[currentImage], 0, sizeof(baseUBO), 0, &data);
        memcpy(data, &baseUBO, sizeof(baseUBO));
    vkUnmapMemory(app->getDevice(), base.sceneUniformBuffersMemory[currentImage]);

    vkMapMemory(app->getDevice(), second.uniformBuffersMemory[currentImage], 0, sizeof(baseUBO), 0, &data);
        memcpy(data, &baseUBO, sizeof(baseUBO));
    vkUnmapMemory(app->getDevice(), second.uniformBuffersMemory[currentImage]);
}

void graphics::updateSkyboxUniformBuffer(uint32_t currentImage)
{
    if(skybox.objects.size()!=0)
    {
        void* data;

        SkyboxUniformBufferObject skyboxUBO{};
            skyboxUBO.view = cameraObject->getViewMatrix();
            skyboxUBO.proj = glm::perspective(glm::radians(45.0f), (float) image.Extent.width / (float) image.Extent.height, 0.1f, 1000.0f);
            skyboxUBO.proj[1][1] *= -1;
            skyboxUBO.model = glm::translate(glm::mat4x4(1.0f),cameraObject->getTranslate())*skybox.objects[0]->ModelMatrix();
        vkMapMemory(app->getDevice(), this->skybox.uniformBuffersMemory[currentImage], 0, sizeof(skyboxUBO), 0, &data);
            memcpy(data, &skyboxUBO, sizeof(skyboxUBO));
        vkUnmapMemory(app->getDevice(), this->skybox.uniformBuffersMemory[currentImage]);
    }
}

void graphics::updateObjectUniformBuffer(uint32_t currentImage)
{
    for(size_t i=0;i<base.objects.size();i++)
        base.objects.at(i)->updateUniformBuffer(currentImage);
    for(size_t i=0;i<bloom.objects.size();i++)
        bloom.objects.at(i)->updateUniformBuffer(currentImage);
    for(size_t i=0;i<oneColor.objects.size();i++)
        oneColor.objects.at(i)->updateUniformBuffer(currentImage);
    for(size_t i=0;i<stencil.objects.size();i++)
        stencil.objects.at(i)->updateUniformBuffer(currentImage);
}

void graphics::bindBaseObject(object *newObject)
{
    base.objects.push_back(newObject);
    base.createObjectDescriptorPool(app,newObject,image.Count);
    base.createObjectDescriptorSet(app,newObject,image.Count,emptyTexture);
}

void graphics::bindBloomObject(object *newObject)
{
    bloom.objects.push_back(newObject);
    bloom.base->createObjectDescriptorPool(app,newObject,image.Count);
    bloom.base->createObjectDescriptorSet(app,newObject,image.Count,emptyTexture);
}

void graphics::bindOneColorObject(object *newObject)
{
    oneColor.objects.push_back(newObject);
    oneColor.base->createObjectDescriptorPool(app,newObject,image.Count);
    oneColor.base->createObjectDescriptorSet(app,newObject,image.Count,emptyTexture);
}

void graphics::bindStencilObject(object *newObject, float lineWidth, glm::vec4 lineColor)
{
    stencil.stencilEnable.push_back(false);
    stencil.stencilWidth.push_back(lineWidth);
    stencil.stencilColor.push_back(lineColor);
    stencil.objects.push_back(newObject);
    stencil.base->createObjectDescriptorPool(app,newObject,image.Count);
    stencil.base->createObjectDescriptorSet(app,newObject,image.Count,emptyTexture);
}

void graphics::bindSkyBoxObject(object *newObject, cubeTexture* texture)
{
    skybox.objects.push_back(newObject);
    skybox.texture = texture;
}

void graphics::removeBinds()
{
    for(auto object: base.objects)
        object->destroyDescriptorPools();
    for(auto object: bloom.objects)
        object->destroyDescriptorPools();
    for(auto object: oneColor.objects)
        object->destroyDescriptorPools();
    for(auto object: stencil.objects)
        object->destroyDescriptorPools();

    base.objects.clear();
    bloom.objects.clear();
    oneColor.objects.clear();
    stencil.objects.clear();
    stencil.stencilEnable.clear();
}

void graphics::setStencilObject(object *oldObject)
{
    for(uint32_t i=0;i<stencil.objects.size();i++)
        if(stencil.objects[i]==oldObject){
            if(stencil.stencilEnable[i] == true)    stencil.stencilEnable[i] = false;
            else                                    stencil.stencilEnable[i] = true;
        }
}

void graphics::bindLightSource(light<spotLight>* object)
{
    second.createLightDescriptorPool(app,object,image.Count);
    second.createLightDescriptorSets(app,object,image.Count);
    updateLightDescriptorSets(object);
}
