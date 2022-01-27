#include "graphics.h"
#include "core/operations.h"
#include "core/transformational/object.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/light.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/camera.h"

graphics::graphics()
{

}

void graphics::setApplication(VkApplication * app){this->app = app;}
void graphics::setEmptyTexture(texture *emptyTexture){this->emptyTexture = emptyTexture;}
void graphics::setMSAASamples(VkSampleCountFlagBits msaaSamples){this->msaaSamples = msaaSamples;}
void graphics::setSkyboxTexture(cubeTexture *tex){skybox.texture = tex;}
void graphics::setImageProp(uint32_t imageCount, VkFormat format, VkExtent2D extent)
{
    this->imageCount = imageCount;
    imageFormat = format;
    this->extent = extent;
}

void graphics::destroy()
{
    base.Destroy(app);
    extension.DestroyBloom(app);
    extension.DestroyGodRays(app);
    skybox.Destroy(app);
    second.Destroy(app);

    vkDestroyRenderPass(app->getDevice(), renderPass, nullptr);
    for(size_t i = 0; i< framebuffers.size();i++)
        vkDestroyFramebuffer(app->getDevice(), framebuffers[i],nullptr);

    depthAttachment.deleteAttachment(&app->getDevice());
    for(size_t i=0;i<colorAttachments.size();i++)
        colorAttachments.at(i).deleteAttachment(&app->getDevice());
    for(size_t i=0;i<Attachments.size();i++)
        Attachments.at(i).deleteAttachment(&app->getDevice());
    for(size_t i=0;i<3;i++)
        Attachments.at(i).deleteSampler(&app->getDevice());
}

//=========================================================================//

void graphics::createAttachments()
{
    if(msaaSamples!=VK_SAMPLE_COUNT_1_BIT)
        createColorAttachments();
    createDepthAttachment();
    createResolveAttachments();
}

void graphics::createColorAttachments()
{
    colorAttachments.resize(9);
    for(size_t i=0;i<3;i++)
    {
        createImage(app,extent.width, extent.height,
                    1, msaaSamples, imageFormat,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorAttachments.at(i).image, colorAttachments.at(i).imageMemory);
        createImageView(app, colorAttachments.at(i).image,
                        imageFormat, VK_IMAGE_ASPECT_COLOR_BIT,
                        1, &colorAttachments.at(i).imageView);
    }
    for(size_t i=3;i<5;i++)
    {
        createImage(app,extent.width, extent.height,
                    1, msaaSamples, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorAttachments.at(i).image, colorAttachments.at(i).imageMemory);
        createImageView(app, colorAttachments.at(i).image,
                        VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
                        1, &colorAttachments.at(i).imageView);
    }
    for(size_t i=5;i<colorAttachments.size();i++)
    {
        createImage(app,extent.width, extent.height,
                    1, msaaSamples, imageFormat,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorAttachments.at(i).image, colorAttachments.at(i).imageMemory);
        createImageView(app, colorAttachments.at(i).image,
                        imageFormat, VK_IMAGE_ASPECT_COLOR_BIT,
                        1, &colorAttachments.at(i).imageView);
    }
}
void graphics::createDepthAttachment()
{
    createImage(app,
                extent.width, extent.height,
                1, msaaSamples,
                findDepthFormat(app),
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                depthAttachment.image, depthAttachment.imageMemory);
    createImageView(app, depthAttachment.image,
                    findDepthFormat(app), VK_IMAGE_ASPECT_DEPTH_BIT,
                    1, &depthAttachment.imageView);
}
void graphics::createResolveAttachments()
{
    Attachments.resize(9);
    for(size_t i=0;i<3;i++)
    {
        Attachments[i].resize(imageCount);
        for(size_t image=0; image<imageCount; image++)
        {
            createImage(app,extent.width,extent.height,
                        1,VK_SAMPLE_COUNT_1_BIT,imageFormat,
                        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Attachments[i].image[image], Attachments[i].imageMemory[image]);
            createImageView(app, Attachments[i].image[image],
                            imageFormat, VK_IMAGE_ASPECT_COLOR_BIT,
                            1, &Attachments[i].imageView[image]);
        }
    }
    for(size_t i=3;i<5;i++)
    {
        Attachments[i].resize(imageCount);
        for(size_t image=0; image<imageCount; image++)
        {
            createImage(app,extent.width,extent.height,
                        1,VK_SAMPLE_COUNT_1_BIT,
                        VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Attachments[i].image[image], Attachments[i].imageMemory[image]);
            createImageView(app, Attachments[i].image[image],
                            VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,
                            1, &Attachments[i].imageView[image]);
        }
    }
    for(size_t i=5;i<Attachments.size();i++)
    {
        Attachments[i].resize(imageCount);
        for(size_t image=0; image<imageCount; image++)
        {
            createImage(app,extent.width,extent.height,
                        1,VK_SAMPLE_COUNT_1_BIT,imageFormat,VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Attachments[i].image[image], Attachments[i].imageMemory[image]);
            createImageView(app, Attachments[i].image[image],
                            imageFormat,VK_IMAGE_ASPECT_COLOR_BIT,
                            1, &Attachments[i].imageView[image]);
        }
    }

    for(size_t i=0;i<3;i++)
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
        {throw std::runtime_error("failed to create texture sampler!");}
    }
}

//=======================================RenderPass======================//

void graphics::createRenderPass()
{
    if(msaaSamples==VK_SAMPLE_COUNT_1_BIT)
    {
        oneSampleRenderPass();
    }else{
        multiSampleRenderPass();
    }
}
    void graphics::oneSampleRenderPass()
    {
        std::vector<VkAttachmentDescription> attachments;
        for(size_t i=0;i<3;i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = imageFormat;
                colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(colorAttachment);
        }
        for(size_t i=3;i<5;i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
                colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
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
                colorAttachment.format = imageFormat;
                colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(colorAttachment);
        }
        VkAttachmentDescription depthAttachment{};
            depthAttachment.format = findDepthFormat(app);
            depthAttachment.samples = msaaSamples;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments.push_back(depthAttachment);

        //===========================first=====================================================//

        uint32_t index = 3;
        std::vector<VkAttachmentReference> firstAttachmentRef(6);
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
        std::vector<VkAttachmentReference> secondInAttachmentRef(6);
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
        index++;
            secondInAttachmentRef.at(index).attachment = 8;
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
            dependency.at(index).srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;                                                           //задаёт как стадии конвейера в исходном проходе создают данные
            dependency.at(index).srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;                                                                     //поля задают как каждый из исходных проходов обращается к данным
            dependency.at(index).dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dependency.at(index).dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        index++;
            dependency.at(index).srcSubpass = 0;                                                                                                //ссылка из исходного прохода
            dependency.at(index).dstSubpass = 1;                                                                                                //в целевой подпроход (поглощающий данные)
            dependency.at(index).srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;                                                  //задаёт как стадии конвейера в исходном проходе создают данные
            dependency.at(index).srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;                    //поля задают как каждый из исходных проходов обращается к данным
            dependency.at(index).dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dependency.at(index).dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

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
        {throw std::runtime_error("failed to create render pass!");}
    }
    void graphics::multiSampleRenderPass()
    {
        std::vector<VkAttachmentDescription> attachments;        
        for(size_t i=0;i<3;i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = imageFormat;
                colorAttachment.samples = msaaSamples;
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
                colorAttachment.samples = msaaSamples;
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
                colorAttachment.format = imageFormat;
                colorAttachment.samples = msaaSamples;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachments.push_back(colorAttachment);
        }

        VkAttachmentDescription depthAttachment{};
            depthAttachment.format = findDepthFormat(app);
            depthAttachment.samples = msaaSamples;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments.push_back(depthAttachment);

        for(size_t i=0;i<3;i++)
        {
            VkAttachmentDescription colorAttachmentResolve{};
                colorAttachmentResolve.format = imageFormat;
                colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments.push_back(colorAttachmentResolve);
        }
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
                colorAttachmentResolve.format = imageFormat;
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
        std::vector<VkAttachmentReference> firstAttachmentRef(6);
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
        std::vector<VkAttachmentReference> firstResolveRef(6);
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
        std::vector<VkAttachmentReference> secondInAttachmentRef(6);
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
        index++;
            secondInAttachmentRef.at(index).attachment = colorAttachments.size()+8;
            secondInAttachmentRef.at(index).layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        index++;
            secondInAttachmentRef.at(index).attachment = colorAttachments.size()+9;
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
        {
            throw std::runtime_error("failed to create render pass!");
        }
    }

//===================Framebuffers===================================

void graphics::createFramebuffers()
{
    if(msaaSamples == VK_SAMPLE_COUNT_1_BIT)
    {
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

        framebuffers.resize(imageCount);
        for (size_t image = 0; image < imageCount; image++)
        {
            std::vector<VkImageView> attachments;
            for(size_t i=0;i<Attachments.size();i++)
                attachments.push_back(Attachments[i].imageView[image]);
            attachments.push_back(depthAttachment.imageView);

            VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;                                                                        //дескриптор объекта прохода рендеринга
                framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());                                    //число изображений
                framebufferInfo.pAttachments = attachments.data();                                                              //набор изображений, которые должны быть привязаны к фреймбуферу, передаётся через массив дескрипторов объектов VkImageView
                framebufferInfo.width = extent.width;                                                                  //ширина изображения
                framebufferInfo.height = extent.height;                                                                //высота изображения
                framebufferInfo.layers = 1;                                                                                     //число слоёв

            if (vkCreateFramebuffer(app->getDevice(), &framebufferInfo, nullptr, &framebuffers[image]) != VK_SUCCESS)  //создание буфера кадров
                throw std::runtime_error("failed to create framebuffer!");
        }
    }
    void graphics::multiSampleFrameBuffer()
    {
        /* Фреймбуфер (буфер кадра) - эо объект, представляющий набор изображений, в который
         * графические конвейеры будут осуществлять рендеринг. Они затрагивают посление несколько
         * стадий в кнвейере: тесты глубины и трафарета, смешивание цветов, логические операции,
         * мультисемплинг и т.п. Фреймбуфер создаётся, используя ссылку на проход рендеринга, и может быть
         * использован с любым проходом рендеринга, имеющим похожую структуру подключений.*/

        framebuffers.resize(imageCount);
        for (size_t image = 0; image < imageCount; image++)
        {
            std::vector<VkImageView> attachments;
            for(size_t i=0;i<colorAttachments.size();i++)
                attachments.push_back(colorAttachments[i].imageView);
            attachments.push_back(depthAttachment.imageView);
            for(size_t i=0;i<Attachments.size();i++)
                attachments.push_back(Attachments[i].imageView[image]);


            VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;                                                                            //дескриптор объекта прохода рендеринга
                framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());                                        //число изображений
                framebufferInfo.pAttachments = attachments.data();                                                                  //набор изображений, которые должны быть привязаны к фреймбуферу, передаётся через массив дескрипторов объектов VkImageView
                framebufferInfo.width = extent.width;                                                                      //ширина изображения
                framebufferInfo.height = extent.height;                                                                    //высота изображения
                framebufferInfo.layers = 1;                                                                                         //число слоёв

            if (vkCreateFramebuffer(app->getDevice(), &framebufferInfo, nullptr, &framebuffers[image]) != VK_SUCCESS)  //создание буфера кадров
                throw std::runtime_error("failed to create framebuffer!");
        }
    }

void graphics::createPipelines()
{
    base.createDescriptorSetLayout(app);
    base.createPipeline(app,{imageCount,extent,msaaSamples,renderPass});
    base.createUniformBuffers(app,imageCount);
    extension.createBloomPipeline(app,&base,{imageCount,extent,msaaSamples,renderPass});
    extension.createGodRaysPipeline(app,&base,{imageCount,extent,msaaSamples,renderPass});
    skybox.createDescriptorSetLayout(app);
    skybox.createPipeline(app,{imageCount,extent,msaaSamples,renderPass});
    skybox.createUniformBuffers(app,imageCount);
    second.createDescriptorSetLayout(app);
    second.createPipeline(app,{imageCount,extent,msaaSamples,renderPass});
    second.createUniformBuffers(app,imageCount);
}

void graphics::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i, std::vector<object*> & object3D, object & sky)
{
    std::array<VkClearValue, 10> clearValues{};
        for(size_t i=0;i<9;i++)
            clearValues[i].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[9].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo drawRenderPassInfo{};
        drawRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        drawRenderPassInfo.renderPass = renderPass;
        drawRenderPassInfo.framebuffer = framebuffers[i];
        drawRenderPassInfo.renderArea.offset = {0, 0};
        drawRenderPassInfo.renderArea.extent = extent;
        drawRenderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        drawRenderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[i], &drawRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        for(size_t j = 0; j<object3D.size() ;j++)
        {
            VkDeviceSize offsets[1] = { 0 };

            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, *object3D[j]->getPipeline());
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, & object3D[j]->getModel()->vertices.buffer, offsets);

            if (object3D[j]->getModel()->indices.buffer != VK_NULL_HANDLE)
                vkCmdBindIndexBuffer(commandBuffers[i],  object3D[j]->getModel()->indices.buffer, 0, VK_INDEX_TYPE_UINT32);


            for (auto node : object3D[j]->getModel()->nodes)
                renderNode(node,commandBuffers[i],base.DescriptorSets[i],object3D[j]->getDescriptorSet()[i],*object3D[j]->getPipelineLayout());

//            for (auto node : object3D[j]->getModel()->nodes)
//               if(glm::length(glm::vec3(object3D[j]->getTransformation()*node->matrix*glm::vec4(0.0f,0.0f,0.0f,1.0f))-cameraPosition)<object3D[j]->getVisibilityDistance())
//                    renderNode(node,commandBuffers[i],base.DescriptorSets[i],object3D[j]->getDescriptorSet()[i],*object3D[j]->getPipelineLayout());
        }

        //skybox
        VkDeviceSize offsets[1] = { 0 };
        vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, & sky.getModel()->vertices.buffer, offsets);
        vkCmdBindIndexBuffer(commandBuffers[i],  sky.getModel()->indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.Pipeline);
        vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.PipelineLayout, 0, 1, &skybox.DescriptorSets[i], 0, NULL);
        vkCmdDrawIndexed(commandBuffers[i], 36, 1, 0, 0, 0);

    vkCmdNextSubpass(commandBuffers[i], VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, second.Pipeline);
        vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, second.PipelineLayout, 0, 1, &second.DescriptorSets[i], 0, nullptr);
        vkCmdDraw(commandBuffers[i], 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers[i]);
}

void graphics::renderNode(Node *node, VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, VkDescriptorSet& objectDescriptorSet, VkPipelineLayout& layout)
{
    if (node->mesh)
    {
        for (Primitive* primitive : node->mesh->primitives)
        {
            const std::vector<VkDescriptorSet> descriptorsets =
            {
                descriptorSet,
                objectDescriptorSet,
                node->mesh->uniformBuffer.descriptorSet,
                primitive->material.descriptorSet
            };
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, static_cast<uint32_t>(descriptorsets.size()), descriptorsets.data(), 0, NULL);

            // Pass material parameters as push constants
            PushConstBlockMaterial pushConstBlockMaterial{};

            pushConstBlockMaterial.emissiveFactor = primitive->material.emissiveFactor;
            // To save push constant space, availabilty and texture coordiante set are combined
            // -1 = texture not used for this material, >= 0 texture used and index of texture coordinate set
            pushConstBlockMaterial.colorTextureSet = primitive->material.baseColorTexture != nullptr ? primitive->material.texCoordSets.baseColor : -1;
            pushConstBlockMaterial.normalTextureSet = primitive->material.normalTexture != nullptr ? primitive->material.texCoordSets.normal : -1;
            pushConstBlockMaterial.occlusionTextureSet = primitive->material.occlusionTexture != nullptr ? primitive->material.texCoordSets.occlusion : -1;
            pushConstBlockMaterial.emissiveTextureSet = primitive->material.emissiveTexture != nullptr ? primitive->material.texCoordSets.emissive : -1;
            pushConstBlockMaterial.alphaMask = static_cast<float>(primitive->material.alphaMode == Material::ALPHAMODE_MASK);
            pushConstBlockMaterial.alphaMaskCutoff = primitive->material.alphaCutoff;

            // TODO: glTF specs states that metallic roughness should be preferred, even if specular glosiness is present

            if (primitive->material.pbrWorkflows.metallicRoughness) {
                // Metallic roughness workflow
                pushConstBlockMaterial.workflow = static_cast<float>(PBR_WORKFLOW_METALLIC_ROUGHNESS);
                pushConstBlockMaterial.baseColorFactor = primitive->material.baseColorFactor;
                pushConstBlockMaterial.metallicFactor = primitive->material.metallicFactor;
                pushConstBlockMaterial.roughnessFactor = primitive->material.roughnessFactor;
                pushConstBlockMaterial.PhysicalDescriptorTextureSet = primitive->material.metallicRoughnessTexture != nullptr ? primitive->material.texCoordSets.metallicRoughness : -1;
                pushConstBlockMaterial.colorTextureSet = primitive->material.baseColorTexture != nullptr ? primitive->material.texCoordSets.baseColor : -1;
            }

            if (primitive->material.pbrWorkflows.specularGlossiness) {
                // Specular glossiness workflow
                pushConstBlockMaterial.workflow = static_cast<float>(PBR_WORKFLOW_SPECULAR_GLOSINESS);
                pushConstBlockMaterial.PhysicalDescriptorTextureSet = primitive->material.extension.specularGlossinessTexture != nullptr ? primitive->material.texCoordSets.specularGlossiness : -1;
                pushConstBlockMaterial.colorTextureSet = primitive->material.extension.diffuseTexture != nullptr ? primitive->material.texCoordSets.baseColor : -1;
                pushConstBlockMaterial.diffuseFactor = primitive->material.extension.diffuseFactor;
                pushConstBlockMaterial.specularFactor = glm::vec4(primitive->material.extension.specularFactor, 1.0f);
            }

            vkCmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstBlockMaterial), &pushConstBlockMaterial);

            if (primitive->hasIndices)
            {
                vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
            }
            else
            {
                vkCmdDraw(commandBuffer, primitive->vertexCount, 1, 0, 0);
            }
        }
    }

    for (auto child : node->children)
        renderNode(child, commandBuffer, descriptorSet,objectDescriptorSet,layout);
}

void graphics::updateUniformBuffer(uint32_t currentImage, camera *cam, object *skybox)
{
    cameraPosition = glm::vec4(cam->getTranslate(), 1.0);

    void* data;

    UniformBufferObject baseUBO{};
        baseUBO.view = cam->getViewMatrix();
        baseUBO.proj = glm::perspective(glm::radians(45.0f), (float) extent.width / (float) extent.height, 0.1f, 1000.0f);
        baseUBO.proj[1][1] *= -1;
        baseUBO.eyePosition = glm::vec4(cam->getTranslate(), 1.0);
    vkMapMemory(app->getDevice(), base.uniformBuffersMemory[currentImage], 0, sizeof(baseUBO), 0, &data);
        memcpy(data, &baseUBO, sizeof(baseUBO));
    vkUnmapMemory(app->getDevice(), base.uniformBuffersMemory[currentImage]);

    SkyboxUniformBufferObject skyboxUBO{};
        skyboxUBO.view = cam->getViewMatrix();
        skyboxUBO.proj = glm::perspective(glm::radians(45.0f), (float) extent.width / (float) extent.height, 0.1f, 1000.0f);
        skyboxUBO.proj[1][1] *= -1;
        skyboxUBO.model = glm::translate(glm::mat4x4(1.0f),cam->getTranslate())*skybox->getTransformation();
    vkMapMemory(app->getDevice(), this->skybox.uniformBuffersMemory[currentImage], 0, sizeof(skyboxUBO), 0, &data);
        memcpy(data, &skyboxUBO, sizeof(skyboxUBO));
    vkUnmapMemory(app->getDevice(), this->skybox.uniformBuffersMemory[currentImage]);

    SecondUniformBufferObject secondUBO{};
        secondUBO.eyePosition = glm::vec4(cam->getTranslate(), 1.0);
    vkMapMemory(app->getDevice(), second.uniformBuffersMemory[currentImage], 0, sizeof(secondUBO), 0, &data);
        memcpy(data, &secondUBO, sizeof(secondUBO));
    vkUnmapMemory(app->getDevice(), second.uniformBuffersMemory[currentImage]);
}

VkPipeline                      &graphics::PipeLine(){return base.Pipeline;}
VkPipeline                      &graphics::BloomSpriteGraphicsPipeline(){return extension.bloomPipeline;}
VkPipeline                      &graphics::GodRaysPipeline(){return extension.godRaysPipeline;}
VkPipeline                      &graphics::SkyBoxPipeLine(){return skybox.Pipeline;}

VkPipelineLayout                &graphics::PipelineLayout(){return base.PipelineLayout;}
VkPipelineLayout                &graphics::BloomSpritePipelineLayout(){return extension.bloomPipelineLayout;}
VkPipelineLayout                &graphics::GodRaysPipelineLayout(){return extension.godRaysPipelineLayout;}
VkPipelineLayout                &graphics::SkyBoxPipelineLayout(){return skybox.PipelineLayout;}

std::vector<attachments>        &graphics::getAttachments(){return Attachments;}
