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

void graphics::setApplication(VkApplication * app)
{
    this->app = app;
}

void graphics::setEmptyTexture(texture *emptyTexture)
{
    this->emptyTexture = emptyTexture;
}

void graphics::setMSAASamples(VkSampleCountFlagBits msaaSamples)
{
    this->msaaSamples = msaaSamples;
}

void graphics::destroy()
{
    depthAttachment.deleteAttachment(&app->getDevice());

    for(size_t i=0;i<colorAttachments.size();i++)
    {
        colorAttachments.at(i).deleteAttachment(&app->getDevice());
    }

    vkDestroyPipeline(app->getDevice(), graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), pipelineLayout,nullptr);

    vkDestroyPipeline(app->getDevice(), bloomSpriteGraphicsPipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), bloomSpritePipelineLayout, nullptr);

    vkDestroyPipeline(app->getDevice(), godRaysPipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), godRaysPipelineLayout,nullptr);

    vkDestroyPipeline(app->getDevice(), skyBox.pipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), skyBox.pipelineLayout,nullptr);

    vkDestroyRenderPass(app->getDevice(), renderPass, nullptr);

    for(size_t i=0;i<Attachments.size();i++)
    {
        Attachments.at(i).deleteAttachment(&app->getDevice());
        Attachments.at(i).deleteSampler(&app->getDevice());
    }

    for(size_t i = 0; i< swapChainFramebuffers.size();i++)
    {
        vkDestroyFramebuffer(app->getDevice(), swapChainFramebuffers[i],nullptr);
    }

    vkDestroyDescriptorSetLayout(app->getDevice(), descriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(app->getDevice(), uniformBufferSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(app->getDevice(), uniformBlockSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(app->getDevice(), materialSetLayout, nullptr);
    vkDestroyDescriptorPool(app->getDevice(), descriptorPool, nullptr);

    vkDestroyDescriptorSetLayout(app->getDevice(), skyBox.descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(app->getDevice(), skyBox.descriptorPool, nullptr);

    for (size_t i = 0; i < imageCount; i++)
    {
        vkDestroyBuffer(app->getDevice(), skyBox.uniformBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), skyBox.uniformBuffersMemory[i], nullptr);

        vkDestroyBuffer(app->getDevice(), uniformBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), uniformBuffersMemory[i], nullptr);

        vkDestroyBuffer(app->getDevice(), emptyUniformBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), emptyUniformBuffersMemory[i], nullptr);
    }

    vkDestroyDescriptorSetLayout(app->getDevice(), secondDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(app->getDevice(), secondDescriptorPool, nullptr);
    vkDestroyPipeline(app->getDevice(), secondGraphicsPipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), secondPipelineLayout, nullptr);
}

void graphics::setImageProp(uint32_t imageCount, VkFormat format, VkExtent2D extent)
{
    this->imageCount = imageCount;
    swapChainImageFormat = format;
    swapChainExtent = extent;
}

void graphics::setSkyboxTexture(cubeTexture *tex)
{
    skyBox.texture = tex;
}

//=========================================================================//

void graphics::createColorAttachments()
{
    if(msaaSamples!=VK_SAMPLE_COUNT_1_BIT)
    {
        colorAttachments.resize(9);

        for(size_t i=0;i<3;i++)
        {
            createImage(app,swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, swapChainImageFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorAttachments.at(i).image, colorAttachments.at(i).imageMemory);
            colorAttachments.at(i).imageView = createImageView(app, colorAttachments.at(i).image, swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
        for(size_t i=3;i<5;i++)
        {
            createImage(app,swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorAttachments.at(i).image, colorAttachments.at(i).imageMemory);
            colorAttachments.at(i).imageView = createImageView(app, colorAttachments.at(i).image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
        for(size_t i=5;i<colorAttachments.size();i++)
        {
            createImage(app,swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, swapChainImageFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorAttachments.at(i).image, colorAttachments.at(i).imageMemory);
            colorAttachments.at(i).imageView = createImageView(app, colorAttachments.at(i).image, swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
    }
}

void graphics::createDepthAttachment()
{
    VkFormat depthFormat = findDepthFormat(app);
    createImage(app, swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthAttachment.image, depthAttachment.imageMemory);
    depthAttachment.imageView = createImageView(app, depthAttachment.image, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
}

void graphics::createAttachments()
{
    Attachments.resize(9);
    for(size_t i=0;i<3;i++)
    {
        Attachments[i].resize(imageCount);
        for(size_t image=0; image<imageCount; image++)
        {
            createImage(app,swapChainExtent.width,swapChainExtent.height,1,VK_SAMPLE_COUNT_1_BIT,swapChainImageFormat,VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Attachments[i].image[image], Attachments[i].imageMemory[image]);
            Attachments[i].imageView[image] = createImageView(app, Attachments[i].image[image], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
    }
    for(size_t i=3;i<5;i++)
    {
        Attachments[i].resize(imageCount);
        for(size_t image=0; image<imageCount; image++)
        {
            createImage(app,swapChainExtent.width,swapChainExtent.height,1,VK_SAMPLE_COUNT_1_BIT,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Attachments[i].image[image], Attachments[i].imageMemory[image]);
            Attachments[i].imageView[image] = createImageView(app, Attachments[i].image[image],VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
    }
    for(size_t i=5;i<Attachments.size();i++)
    {
        Attachments[i].resize(imageCount);
        for(size_t image=0; image<imageCount; image++)
        {
            createImage(app,swapChainExtent.width,swapChainExtent.height,1,VK_SAMPLE_COUNT_1_BIT,swapChainImageFormat,VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Attachments[i].image[image], Attachments[i].imageMemory[image]);
            Attachments[i].imageView[image] = createImageView(app, Attachments[i].image[image],swapChainImageFormat,VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
    }

    for(size_t i=0;i<Attachments.size();i++)
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

void graphics::createUniformBuffers()
{
    /* Uniform - блоки представляют быстрый доступ к константным (доступным только для чтения) данным, хранимым в буферах
     * Они объявляются как структуры в шейдере и прикрепляются к памяти при помощи русурса буфера, привяанного ко множеству дескрипторов*/

    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(imageCount);
    uniformBuffersMemory.resize(imageCount);

    for (size_t i = 0; i < imageCount; i++)
    {
        createBuffer(app,bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
    }

    emptyUniformBuffers.resize(imageCount);
    emptyUniformBuffersMemory.resize(imageCount);

    for (size_t i = 0; i < imageCount; i++)
    {
        createBuffer(app,sizeof(LightUniformBufferObject),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, emptyUniformBuffers[i], emptyUniformBuffersMemory[i]);
    }
}

void graphics::createSkyboxUniformBuffers()
{
    VkDeviceSize bufferSize = sizeof(SkyboxUniformBufferObject);

    skyBox.uniformBuffers.resize(imageCount);
    skyBox.uniformBuffersMemory.resize(imageCount);

    for (size_t i = 0; i < imageCount; i++)
    {
        createBuffer(app,bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, skyBox.uniformBuffers[i], skyBox.uniformBuffersMemory[i]);
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
                colorAttachment.format = swapChainImageFormat;
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
                colorAttachment.format = swapChainImageFormat;
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
            dependency.at(index).srcSubpass = VK_SUBPASS_EXTERNAL;                                                                                //ссылка из исходного прохода (создавшего данные)
            dependency.at(index).dstSubpass = 0;                                                                                                  //в целевой подпроход (поглощающий данные)
            dependency.at(index).srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;       //задаёт как стадии конвейера в исходном проходе создают данные
            dependency.at(index).srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;                                                                                               //поля задают как каждый из исходных проходов обращается к данным
            dependency.at(index).dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dependency.at(index).dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        index++;
            dependency.at(index).srcSubpass = 0;                                                                                //ссылка из исходного прохода (создавшего данные)
            dependency.at(index).dstSubpass = 1;                                                                                                  //в целевой подпроход (поглощающий данные)
            dependency.at(index).srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;       //задаёт как стадии конвейера в исходном проходе создают данные
            dependency.at(index).srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;                                                                                               //поля задают как каждый из исходных проходов обращается к данным
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
        {
            throw std::runtime_error("failed to create render pass!");
        }
    }
    void graphics::multiSampleRenderPass()
    {
        /*  Одной из вещей, отличающий графический конвейер Vulkan от вычислительного
         * конвейера, является то, что вы используете графический конвейер для рендерига
         * пикселов в изображения, которые вы либо дальше будете обрабатывать, либо
         * покажете пользователю. В сложгых графических приложения изображение строится из
         * нескольких прохов, где каждый проход отвечает ха построение отдельной части сцены,
         * применяя полноэкранные эффекты, такие как постпроцессинг или наложение (composition),
         * рендерин пользовательского интерфейса и т.п.
         *  Эти проходы могут быть представлены в Vulkan при помощи объектов прохода рендеринга (renderpass)
         * Один объект прохода рендеринга совмещаеь в себе несколько проходов или фаз рендеринга над набором
         * выходных изображений. Каждый проход внутри такого прохода рендеринга называется подпроходом (subpass)
         * Объекты прохода рендеринга могут содержать много проходов, но даже в простом приложении единственным проходом
         * над одним выходным изображением объект прохода рендеринга содержит информацию об этом выходном изображении.
         *  Весь рендеринг должен содержаться внутри прохода рендеринга. Более того, графические конвейеры должны знать, куда они осуществляют рендеринг;
         * поэтому необэодимо создать объект прохода рендеринга перед созданием графического конвейера, чтобы мы могли сообщить конвейеру о тех изображениях,
         * которые он будет создавать.*/

        std::vector<VkAttachmentDescription> attachments;        
        for(size_t i=0;i<3;i++)
        {
            VkAttachmentDescription colorAttachment{};
                colorAttachment.format = swapChainImageFormat;
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
                colorAttachment.format = swapChainImageFormat;
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
                colorAttachmentResolve.format = swapChainImageFormat;
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
                colorAttachmentResolve.format = swapChainImageFormat;
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
            dependency.at(index).srcSubpass = VK_SUBPASS_EXTERNAL;                                                                                //ссылка из исходного прохода (создавшего данные)
            dependency.at(index).dstSubpass = 0;                                                                                                  //в целевой подпроход (поглощающий данные)
            dependency.at(index).srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;       //задаёт как стадии конвейера в исходном проходе создают данные
            dependency.at(index).srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;                                                                                               //поля задают как каждый из исходных проходов обращается к данным
            dependency.at(index).dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dependency.at(index).dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        index++;
            dependency.at(index).srcSubpass = 0;                                                                                //ссылка из исходного прохода (создавшего данные)
            dependency.at(index).dstSubpass = 1;                                                                                                  //в целевой подпроход (поглощающий данные)
            dependency.at(index).srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;       //задаёт как стадии конвейера в исходном проходе создают данные
            dependency.at(index).srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;                                                                                               //поля задают как каждый из исходных проходов обращается к данным
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
        {
            throw std::runtime_error("failed to create render pass!");
        }
    }

//==========================BaseDescriptors============================

void graphics::createDescriptorSetLayout()
{
    /* Нам нужно предоставить подробную информацию о каждой привязке дескриптора,
     * используемой в шейдерах для создания конвейера, точно так же, как мы должны
     * были сделать для каждого атрибута вершины и ее locationиндекса. Мы создадим
     * новую функцию для определения всей этой информации с именем createDescriptorSetLayout*/

    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    std::array<VkDescriptorSetLayoutBinding, 1> bindings = {uboLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    VkDescriptorSetLayoutBinding uniformBufferLayoutBinding{};
    uniformBufferLayoutBinding.binding = 0;
    uniformBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBufferLayoutBinding.descriptorCount = 1;
    uniformBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uniformBufferLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo uniformBufferLayoutInfo{};
    uniformBufferLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    uniformBufferLayoutInfo.bindingCount = 1;
    uniformBufferLayoutInfo.pBindings = &uniformBufferLayoutBinding;

    if (vkCreateDescriptorSetLayout(app->getDevice(), &uniformBufferLayoutInfo, nullptr, &uniformBufferSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    VkDescriptorSetLayoutBinding uniformBlockLayoutBinding{};
    uniformBlockLayoutBinding.binding = 0;
    uniformBlockLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBlockLayoutBinding.descriptorCount = 1;
    uniformBlockLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uniformBlockLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo uniformBlockLayoutInfo{};
    uniformBlockLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    uniformBlockLayoutInfo.bindingCount = 1;
    uniformBlockLayoutInfo.pBindings = &uniformBlockLayoutBinding;

    if (vkCreateDescriptorSetLayout(app->getDevice(), &uniformBlockLayoutInfo, nullptr, &uniformBlockSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    VkDescriptorSetLayoutBinding baseColorTexture{};
    baseColorTexture.binding = 0;
    baseColorTexture.descriptorCount = 1;
    baseColorTexture.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    baseColorTexture.pImmutableSamplers = nullptr;
    baseColorTexture.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding metallicRoughnessTexture{};
    metallicRoughnessTexture.binding = 1;
    metallicRoughnessTexture.descriptorCount = 1;
    metallicRoughnessTexture.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    metallicRoughnessTexture.pImmutableSamplers = nullptr;
    metallicRoughnessTexture.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding normalTexture{};
    normalTexture.binding = 2;
    normalTexture.descriptorCount = 1;
    normalTexture.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    normalTexture.pImmutableSamplers = nullptr;
    normalTexture.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding occlusionTexture{};
    occlusionTexture.binding = 3;
    occlusionTexture.descriptorCount = 1;
    occlusionTexture.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    occlusionTexture.pImmutableSamplers = nullptr;
    occlusionTexture.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding emissiveTexture{};
    emissiveTexture.binding = 4;
    emissiveTexture.descriptorCount = 1;
    emissiveTexture.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    emissiveTexture.pImmutableSamplers = nullptr;
    emissiveTexture.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 5> materialLayoutBinding={baseColorTexture,metallicRoughnessTexture,normalTexture,occlusionTexture,emissiveTexture};
    VkDescriptorSetLayoutCreateInfo materialLayoutInfo{};
    materialLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    materialLayoutInfo.bindingCount = static_cast<uint32_t>(materialLayoutBinding.size());
    materialLayoutInfo.pBindings = materialLayoutBinding.data();

    if (vkCreateDescriptorSetLayout(app->getDevice(), &materialLayoutInfo, nullptr, &materialSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void graphics::createDescriptorPool(const std::vector<object*> & object3D)
{
    /* Наборы дескрипторов нельзя создавать напрямую, они должны выделяться из пула, как буферы команд.
     * Эквивалент для наборов дескрипторов неудивительно называется пулом дескрипторов . Мы напишем
     * новую функцию createDescriptorPool для ее настройки.*/

    {
        std::vector<VkDescriptorPoolSize> poolSizes(1);
        size_t index = 0;
            poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;                           //Сначала нам нужно описать, какие типы дескрипторов будут содержать наши наборы дескрипторов
            poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);                //и сколько их, используя VkDescriptorPoolSizeструктуры.
        index++;

        //Мы будем выделять один из этих дескрипторов для каждого кадра. На эту структуру размера пула ссылается главный VkDescriptorPoolCreateInfo:
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(imageCount);

        if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    for(size_t i=0;i<object3D.size();i++)
    {
        object3D.at(i)->setDescriptorSetLayouts({&uniformBufferSetLayout,&uniformBlockSetLayout,&materialSetLayout});
        object3D.at(i)->createDescriptorPool(imageCount);
    }
}

void graphics::createDescriptorSets(const std::vector<object*> & object3D)
{
    //Теперь мы можем выделить сами наборы дескрипторов
    /* В нашем случае мы создадим один набор дескрипторов для каждого изображения цепочки подкачки, все с одинаковым макетом.
     * К сожалению, нам нужны все копии макета, потому что следующая функция ожидает массив, соответствующий количеству наборов.
     * Добавьте член класса для хранения дескрипторов набора дескрипторов и назначьте их vkAllocateDescriptorSets */

    for(size_t i=0;i<object3D.size();i++)
    {
        object3D.at(i)->createDescriptorSet(imageCount);
    }

    {
        std::vector<VkDescriptorSetLayout> layouts(imageCount, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(imageCount);
        if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, descriptorSets.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        //Наборы дескрипторов уже выделены, но дескрипторы внутри еще нуждаются в настройке.
        //Теперь мы добавим цикл для заполнения каждого дескриптора:
        for (size_t i = 0; i < imageCount; i++)
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            std::array<VkWriteDescriptorSet, 1> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }
}

//==========================SkyboxDescriptors============================

void graphics::createSkyboxDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding textursBinding{};
    textursBinding.binding = 1;
    textursBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    textursBinding.descriptorCount = 1;
    textursBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    textursBinding.pImmutableSamplers = nullptr;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding,textursBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &skyBox.descriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void graphics::createSkyboxDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes(2);
    size_t index = 0;

    poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;                           //Сначала нам нужно описать, какие типы дескрипторов будут содержать наши наборы дескрипторов
    poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);                //и сколько их, используя VkDescriptorPoolSizeструктуры.
    index++;

    poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;

    //Мы будем выделять один из этих дескрипторов для каждого кадра. На эту структуру размера пула ссылается главный VkDescriptorPoolCreateInfo:
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(imageCount);

    if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &skyBox.descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void graphics::createSkyboxDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> layouts(imageCount, skyBox.descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = skyBox.descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
    allocInfo.pSetLayouts = layouts.data();

    skyBox.descriptorSets.resize(imageCount);
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, skyBox.descriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    //Наборы дескрипторов уже выделены, но дескрипторы внутри еще нуждаются в настройке.
    //Теперь мы добавим цикл для заполнения каждого дескриптора:
    for (size_t i = 0; i < imageCount; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = skyBox.uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(SkyboxUniformBufferObject);

        VkDescriptorImageInfo skyBoxImageInfo;
        skyBoxImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        skyBoxImageInfo.imageView = skyBox.texture->getTextureImageView();
        skyBoxImageInfo.sampler = skyBox.texture->getTextureSampler();

        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = skyBox.descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = skyBox.descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &skyBoxImageInfo;

        vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

//==============================Piplines=================================

void graphics::createPipelines()
{
    createGraphicsPipeline();
    createBloomSpriteGraphicsPipeline();
    createGodRaysGraphicsPipeline();
    createSkyBoxPipeline();
}

void graphics::createGraphicsPipeline()
{
    //считываем шейдеры
    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\base\\basevert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\base\\basefrag.spv");
    //создаём шейдерные модули
    VkShaderModule vertShaderModule = createShaderModule(app, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(app, fragShaderCode);
    //задаём стадии шейдеров в конвейере
    //вершинный
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;                             //ниформацию о всех битах смотри на странице 222
    vertShaderStageInfo.module = vertShaderModule;                                      //сюда передаём шейдерный модуль
    vertShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа
    //фрагментный
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;                           //ниформацию о всех битах смотри на странице 222
    fragShaderStageInfo.module = fragShaderModule;                                      //сюда передаём шейдерный модуль
    fragShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа
    //формаируем нужный массив, который будем передавать в структуру для создания графического конвейера
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    /* Для рендеринга настоящей геометрии вам необходимо передавать данные в конвайер Vulkan.
     * Вы можете использовать индексы вершин и экземпляров, доступные в SPIR-V, для автоматической
     * генерации геометрии или же явно извлекать геометрические данные из буфера. Вместо этого вы можете
     * описать размещение геометрических данных в памяти, и Vulkan может сам извлекать эти данные для вас, передавая их прямо в шейдер*/

    auto bindingDescription = gltfModel::Vertex::getBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;                                                      //количество привязанных дескрипторов вершин
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());  //количество дескрипторов атрибутов вершин
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;                                       //указатель на массив соответствующийх структуру
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();                            //указатель на массив соответствующийх структуру

    /* фаза входной сборки графического конвейера берёт данные в вершинах и группирует их в примитивы,
     * готовые для обработки следубщими стадиями конвейера.*/

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;                       //тип примитива, подробно про тип примитива смотри со страницы 228
    inputAssembly.primitiveRestartEnable = VK_FALSE;                                    //это флаг, кторый используется для того, чтоюы можно было оборвать примитивы полосы и веера и затем начать их снова
                                                                                        //без него кажда полоса и веер потребуют отдельной команды вывода.

    /* здесь может быть добавлена тесселяция*/


    /* Преобразование области вывода - это последнее преобразование координат в конвейере Vulkan до растретизации.
     * Оно преобразует координаты вершины из нормализованных координат устройства в оконные координаты. Одновременно
     * может использоваться несколько областей вывода.*/

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) swapChainExtent.width;
    viewport.height = (float) swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;                                                //число областей вывода
    viewportState.pViewports = &viewport;                                           //размер каждой области вывода
    viewportState.scissorCount = 1;                                                 //число прямоугольников
    viewportState.pScissors = &scissor;                                             //эксцент

    /* Растеризация - это процесс, в ходе которого примитивы, представленные вершинами, преобразуются в потоки фрагментов, которых к обработке
     * фрагментным шейдером. Состояние растеризации управляется тем, как этот процесс происходит, и задаётся при помощи следующей структуры*/

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;                                      //используется для того чтобы полностью выключить растеризацию. Когда флаг установлен, растеризация не работает и не создаются фрагменты
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;                                      //используется для того чтобы Vulkan автоматически превращал треугольники в точки или отрезки
    rasterizer.lineWidth = 1.0f;                                                        //толщина линии
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;                                        //параметр обрасывания
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;                             //параметр направления обхода (против часовой стрелки)
    rasterizer.depthBiasEnable = VK_FALSE;                                              //используется для того чтобы включать отсечение глубины
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional                              //
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    /* Мультсемплинг - это процесс создания нескольких образцов (sample) для каждого пиксела в изображении.
     * Они используются для борьбы с алиансингом и может заметно улучшить общее качество изображения при эффективном использовании*/

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = msaaSamples;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional

    /* Последней стадией в графическом конвейере является стадия смешивания цветов. Эта стадия отвечает за запись фрагментов
     * в цветовые подключения. Во многих случаях это простая операция, которая просто записывает содержимое выходного значения
     * фрагментного шейдера поверх старого значения. Однакоподдеживаются смешивание этих значнеий со значениями,
     * уже находящимися во фрейм буфере, и выполнение простых логических операций между выходными значениями фрагментного
     * шейдера и текущим содержанием фреймбуфера.*/

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment(6);
    for(uint32_t index=0;index<colorBlendAttachment.size();index++)
    {
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_FALSE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].alphaBlendOp = VK_BLEND_OP_MAX;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;                                                 //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
    colorBlending.logicOp = VK_LOGIC_OP_COPY;                                               //Optional
    colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachment.size());     //количество подключений
    colorBlending.pAttachments = colorBlendAttachment.data();                               //массив подключений
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional

    /* Для того чтобы сделать небольште изменения состояния более удобными, Vulkan предоставляет возможность помечать
     * определенные части графического конвейера как динамически, что значит что они могут быть изменены прямо на месте
     * при помощи команд прямо внутри командного буфера*/

    // добавлено
        VkPushConstantRange pushConstantRange;
        pushConstantRange.stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstBlockMaterial);

    std::array<VkDescriptorSetLayout,4> SetLayouts = {descriptorSetLayout,uniformBufferSetLayout,uniformBufferSetLayout,materialSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;                                            //число структур в массиве структур
    pipelineInfo.pStages = shaderStages;                                    //указывает на массив структур VkPipelineShaderStageCreateInfo, каждая из которых описыват одну стадию
    pipelineInfo.pVertexInputState = &vertexInputInfo;                      //вершинный ввод
    pipelineInfo.pInputAssemblyState = &inputAssembly;                      //фаза входной сборки
    pipelineInfo.pViewportState = &viewportState;                           //Преобразование области вывода
    pipelineInfo.pRasterizationState = &rasterizer;                         //растеризация
    pipelineInfo.pMultisampleState = &multisampling;                        //мультсемплинг
    pipelineInfo.pColorBlendState = &colorBlending;                         //смешивание цветов
    pipelineInfo.layout = pipelineLayout;                                   //
    pipelineInfo.renderPass = renderPass;                                   //проход рендеринга
    pipelineInfo.subpass = 0;                                               //подпроход рендеригка
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &depthStencil;

    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::createBloomSpriteGraphicsPipeline()
{
    //считываем шейдеры
    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\bloomSprite\\vertBloomSprite.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\bloomSprite\\fragBloomSprite.spv");
    //создаём шейдерные модули
    VkShaderModule vertShaderModule = createShaderModule(app, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(app, fragShaderCode);
    //задаём стадии шейдеров в конвейере
    //вершинный
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;                             //ниформацию о всех битах смотри на странице 222
    vertShaderStageInfo.module = vertShaderModule;                                      //сюда передаём шейдерный модуль
    vertShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа
    //фрагментный
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;                           //ниформацию о всех битах смотри на странице 222
    fragShaderStageInfo.module = fragShaderModule;                                      //сюда передаём шейдерный модуль
    fragShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа
    //формаируем нужный массив, который будем передавать в структуру для создания графического конвейера
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    /* Для рендеринга настоящей геометрии вам необходимо передавать данные в конвайер Vulkan.
     * Вы можете использовать индексы вершин и экземпляров, доступные в SPIR-V, для автоматической
     * генерации геометрии или же явно извлекать геометрические данные из буфера. Вместо этого вы можете
     * описать размещение геометрических данных в памяти, и Vulkan может сам извлекать эти данные для вас, передавая их прямо в шейдер*/

    auto bindingDescription = gltfModel::Vertex::getBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;                                                      //количество привязанных дескрипторов вершин
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());  //количество дескрипторов атрибутов вершин
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;                                       //указатель на массив соответствующийх структуру
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();                            //указатель на массив соответствующийх структуру

    /* фаза входной сборки графического конвейера берёт данные в вершинах и группирует их в примитивы,
     * готовые для обработки следубщими стадиями конвейера.*/

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;                       //тип примитива, подробно про тип примитива смотри со страницы 228
    inputAssembly.primitiveRestartEnable = VK_FALSE;                                    //это флаг, кторый используется для того, чтоюы можно было оборвать примитивы полосы и веера и затем начать их снова
                                                                                    //без него кажда полоса и веер потребуют отдельной команды вывода.
    /* здесь может быть добавлена тесселяция*/

    /* Преобразование области вывода - это последнее преобразование координат в конвейере Vulkan до растретизации.
     * Оно преобразует координаты вершины из нормализованных координат устройства в оконные координаты. Одновременно
     * может использоваться несколько областей вывода.*/

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) swapChainExtent.width;
    viewport.height = (float) swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;                                                //число областей вывода
    viewportState.pViewports = &viewport;                                           //размер каждой области вывода
    viewportState.scissorCount = 1;                                                 //число прямоугольников
    viewportState.pScissors = &scissor;                                             //эксцент

    /* Растеризация - это процесс, в ходе которого примитивы, представленные вершинами, преобразуются в потоки фрагментов, которых к обработке
     * фрагментным шейдером. Состояние растеризации управляется тем, как этот процесс происходит, и задаётся при помощи следующей структуры*/

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;                                      //используется для того чтобы полностью выключить растеризацию. Когда флаг установлен, растеризация не работает и не создаются фрагменты
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;                                      //используется для того чтобы Vulkan автоматически превращал треугольники в точки или отрезки
    rasterizer.lineWidth = 1.0f;                                                        //толщина линии
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;                                        //параметр обрасывания
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;                                     //параметр направления обхода (против часовой стрелки)
    rasterizer.depthBiasEnable = VK_FALSE;                                              //используется для того чтобы включать отсечение глубины
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional                              //
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    /* Мультсемплинг - это процесс создания нескольких образцов (sample) для каждого пиксела в изображении.
     * Они используются для борьбы с алиансингом и может заметно улучшить общее качество изображения при эффективном использовании*/

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = msaaSamples;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional

    /* Последней стадией в графическом конвейере является стадия смешивания цветов. Эта стадия отвечает за запись фрагментов
     * в цветовые подключения. Во многих случаях это простая операция, которая просто записывает содержимое выходного значения
     * фрагментного шейдера поверх старого значения. Однакоподдеживаются смешивание этих значнеий со значениями,
     * уже находящимися во фрейм буфере, и выполнение простых логических операций между выходными значениями фрагментного
     * шейдера и текущим содержанием фреймбуфера.*/

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment(6);
    for(uint32_t index=0;index<colorBlendAttachment.size();index++)
    {
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_FALSE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].alphaBlendOp = VK_BLEND_OP_MAX;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;                                                 //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
    colorBlending.logicOp = VK_LOGIC_OP_COPY;                                               //Optional
    colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachment.size());     //количество подключений
    colorBlending.pAttachments = colorBlendAttachment.data();                               //массив подключений
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional

    /* Для того чтобы сделать небольште изменения состояния более удобными, Vulkan предоставляет возможность помечать
     * определенные части графического конвейера как динамически, что значит что они могут быть изменены прямо на месте
     * при помощи команд прямо внутри командного буфера*/

    // добавлено
    VkPushConstantRange pushConstantRange;
    pushConstantRange.stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstBlockMaterial);

    std::array<VkDescriptorSetLayout,4> SetLayouts = {descriptorSetLayout,uniformBufferSetLayout,uniformBufferSetLayout,materialSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional

    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &bloomSpritePipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;                                            //число структур в массиве структур
    pipelineInfo.pStages = shaderStages;                                    //указывает на массив структур VkPipelineShaderStageCreateInfo, каждая из которых описыват одну стадию
    pipelineInfo.pVertexInputState = &vertexInputInfo;                      //вершинный ввод
    pipelineInfo.pInputAssemblyState = &inputAssembly;                      //фаза входной сборки
    pipelineInfo.pViewportState = &viewportState;                           //Преобразование области вывода
    pipelineInfo.pRasterizationState = &rasterizer;                         //растеризация
    pipelineInfo.pMultisampleState = &multisampling;                        //мультсемплинг
    pipelineInfo.pColorBlendState = &colorBlending;                         //смешивание цветов
    pipelineInfo.layout = bloomSpritePipelineLayout;                        //
    pipelineInfo.renderPass = renderPass;                                   //проход рендеринга
    pipelineInfo.subpass = 0;                                               //подпроход рендеригка
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &depthStencil;

    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &bloomSpriteGraphicsPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::createGodRaysGraphicsPipeline()
{
    //считываем шейдеры
    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\godRays\\godRaysVert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\godRays\\godRaysFrag.spv");
    //создаём шейдерные модули
    VkShaderModule vertShaderModule = createShaderModule(app, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(app, fragShaderCode);
    //задаём стадии шейдеров в конвейере
    //вершинный
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;                             //ниформацию о всех битах смотри на странице 222
    vertShaderStageInfo.module = vertShaderModule;                                      //сюда передаём шейдерный модуль
    vertShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа
    //фрагментный
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;                           //ниформацию о всех битах смотри на странице 222
    fragShaderStageInfo.module = fragShaderModule;                                      //сюда передаём шейдерный модуль
    fragShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа
    //формаируем нужный массив, который будем передавать в структуру для создания графического конвейера
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    /* Для рендеринга настоящей геометрии вам необходимо передавать данные в конвайер Vulkan.
     * Вы можете использовать индексы вершин и экземпляров, доступные в SPIR-V, для автоматической
     * генерации геометрии или же явно извлекать геометрические данные из буфера. Вместо этого вы можете
     * описать размещение геометрических данных в памяти, и Vulkan может сам извлекать эти данные для вас, передавая их прямо в шейдер*/

    auto bindingDescription = gltfModel::Vertex::getBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;                                                      //количество привязанных дескрипторов вершин
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());  //количество дескрипторов атрибутов вершин
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;                                       //указатель на массив соответствующийх структуру
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();                            //указатель на массив соответствующийх структуру

    /* фаза входной сборки графического конвейера берёт данные в вершинах и группирует их в примитивы,
     * готовые для обработки следубщими стадиями конвейера.*/

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;                       //тип примитива, подробно про тип примитива смотри со страницы 228
    inputAssembly.primitiveRestartEnable = VK_FALSE;                                    //это флаг, кторый используется для того, чтоюы можно было оборвать примитивы полосы и веера и затем начать их снова
                                                                                    //без него кажда полоса и веер потребуют отдельной команды вывода.

    /* здесь может быть добавлена тесселяция*/


    /* Преобразование области вывода - это последнее преобразование координат в конвейере Vulkan до растретизации.
     * Оно преобразует координаты вершины из нормализованных координат устройства в оконные координаты. Одновременно
     * может использоваться несколько областей вывода.*/

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) swapChainExtent.width;
    viewport.height = (float) swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;                                                //число областей вывода
    viewportState.pViewports = &viewport;                                           //размер каждой области вывода
    viewportState.scissorCount = 1;                                                 //число прямоугольников
    viewportState.pScissors = &scissor;                                             //эксцент

    /* Растеризация - это процесс, в ходе которого примитивы, представленные вершинами, преобразуются в потоки фрагментов, которых к обработке
     * фрагментным шейдером. Состояние растеризации управляется тем, как этот процесс происходит, и задаётся при помощи следующей структуры*/

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;                                      //используется для того чтобы полностью выключить растеризацию. Когда флаг установлен, растеризация не работает и не создаются фрагменты
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;                                      //используется для того чтобы Vulkan автоматически превращал треугольники в точки или отрезки
    rasterizer.lineWidth = 1.0f;                                                        //толщина линии
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;                                        //параметр обрасывания
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;                             //параметр направления обхода (против часовой стрелки)
    rasterizer.depthBiasEnable = VK_FALSE;                                              //используется для того чтобы включать отсечение глубины
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional                              //
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    /* Мультсемплинг - это процесс создания нескольких образцов (sample) для каждого пиксела в изображении.
     * Они используются для борьбы с алиансингом и может заметно улучшить общее качество изображения при эффективном использовании*/

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = msaaSamples;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional

    /* Последней стадией в графическом конвейере является стадия смешивания цветов. Эта стадия отвечает за запись фрагментов
     * в цветовые подключения. Во многих случаях это простая операция, которая просто записывает содержимое выходного значения
     * фрагментного шейдера поверх старого значения. Однакоподдеживаются смешивание этих значнеий со значениями,
     * уже находящимися во фрейм буфере, и выполнение простых логических операций между выходными значениями фрагментного
     * шейдера и текущим содержанием фреймбуфера.*/

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment(6);
    for(uint32_t index=0;index<colorBlendAttachment.size();index++)
    {
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_FALSE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].alphaBlendOp = VK_BLEND_OP_MAX;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;                                                 //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
    colorBlending.logicOp = VK_LOGIC_OP_COPY;                                               //Optional
    colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachment.size());     //количество подключений
    colorBlending.pAttachments = colorBlendAttachment.data();                               //массив подключений
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional


    /* Для того чтобы сделать небольште изменения состояния более удобными, Vulkan предоставляет возможность помечать
     * определенные части графического конвейера как динамически, что значит что они могут быть изменены прямо на месте
     * при помощи команд прямо внутри командного буфера*/

    // добавлено
        VkPushConstantRange pushConstantRange;
        pushConstantRange.stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstBlockMaterial);

    std::array<VkDescriptorSetLayout,4> SetLayouts = {descriptorSetLayout,uniformBufferSetLayout,uniformBufferSetLayout,materialSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &godRaysPipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;                                            //число структур в массиве структур
    pipelineInfo.pStages = shaderStages;                                    //указывает на массив структур VkPipelineShaderStageCreateInfo, каждая из которых описыват одну стадию
    pipelineInfo.pVertexInputState = &vertexInputInfo;                      //вершинный ввод
    pipelineInfo.pInputAssemblyState = &inputAssembly;                      //фаза входной сборки
    pipelineInfo.pViewportState = &viewportState;                           //Преобразование области вывода
    pipelineInfo.pRasterizationState = &rasterizer;                         //растеризация
    pipelineInfo.pMultisampleState = &multisampling;                        //мультсемплинг
    pipelineInfo.pColorBlendState = &colorBlending;                         //смешивание цветов
    pipelineInfo.layout = godRaysPipelineLayout;
    pipelineInfo.renderPass = renderPass;                                   //проход рендеринга
    pipelineInfo.subpass = 0;                                               //подпроход рендеригка
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &depthStencil;

    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &godRaysPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::createSkyBoxPipeline()
{
    //считываем шейдеры
    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\skybox\\skyboxVert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\skybox\\skyboxFrag.spv");
    //создаём шейдерные модули
    VkShaderModule vertShaderModule = createShaderModule(app, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(app, fragShaderCode);
    //задаём стадии шейдеров в конвейере
    //вершинный
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;                             //ниформацию о всех битах смотри на странице 222
    vertShaderStageInfo.module = vertShaderModule;                                      //сюда передаём шейдерный модуль
    vertShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа
    //фрагментный
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;                           //ниформацию о всех битах смотри на странице 222
    fragShaderStageInfo.module = fragShaderModule;                                      //сюда передаём шейдерный модуль
    fragShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа
    //формаируем нужный массив, который будем передавать в структуру для создания графического конвейера
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    /* Для рендеринга настоящей геометрии вам необходимо передавать данные в конвайер Vulkan.
     * Вы можете использовать индексы вершин и экземпляров, доступные в SPIR-V, для автоматической
     * генерации геометрии или же явно извлекать геометрические данные из буфера. Вместо этого вы можете
     * описать размещение геометрических данных в памяти, и Vulkan может сам извлекать эти данные для вас, передавая их прямо в шейдер*/

    auto bindingDescription = gltfModel::Vertex::getBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;                                                      //количество привязанных дескрипторов вершин
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());  //количество дескрипторов атрибутов вершин
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;                                       //указатель на массив соответствующийх структуру
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();                            //указатель на массив соответствующийх структуру

    /* фаза входной сборки графического конвейера берёт данные в вершинах и группирует их в примитивы,
     * готовые для обработки следубщими стадиями конвейера.*/

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;                       //тип примитива, подробно про тип примитива смотри со страницы 228
    inputAssembly.primitiveRestartEnable = VK_FALSE;                                    //это флаг, кторый используется для того, чтоюы можно было оборвать примитивы полосы и веера и затем начать их снова
                                                                                        //без него кажда полоса и веер потребуют отдельной команды вывода.

    /* здесь может быть добавлена тесселяция*/


    /* Преобразование области вывода - это последнее преобразование координат в конвейере Vulkan до растретизации.
     * Оно преобразует координаты вершины из нормализованных координат устройства в оконные координаты. Одновременно
     * может использоваться несколько областей вывода.*/

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) swapChainExtent.width;
    viewport.height = (float) swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;                                                //число областей вывода
    viewportState.pViewports = &viewport;                                           //размер каждой области вывода
    viewportState.scissorCount = 1;                                                 //число прямоугольников
    viewportState.pScissors = &scissor;                                             //эксцент

    /* Растеризация - это процесс, в ходе которого примитивы, представленные вершинами, преобразуются в потоки фрагментов, которых к обработке
     * фрагментным шейдером. Состояние растеризации управляется тем, как этот процесс происходит, и задаётся при помощи следующей структуры*/

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;                                      //используется для того чтобы полностью выключить растеризацию. Когда флаг установлен, растеризация не работает и не создаются фрагменты
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;                                      //используется для того чтобы Vulkan автоматически превращал треугольники в точки или отрезки
    rasterizer.lineWidth = 1.0f;                                                        //толщина линии
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;                                        //параметр обрасывания
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;                             //параметр направления обхода (против часовой стрелки)
    rasterizer.depthBiasEnable = VK_FALSE;                                              //используется для того чтобы включать отсечение глубины
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional                              //
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    /* Мультсемплинг - это процесс создания нескольких образцов (sample) для каждого пиксела в изображении.
     * Они используются для борьбы с алиансингом и может заметно улучшить общее качество изображения при эффективном использовании*/

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = msaaSamples;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional

    /* Последней стадией в графическом конвейере является стадия смешивания цветов. Эта стадия отвечает за запись фрагментов
     * в цветовые подключения. Во многих случаях это простая операция, которая просто записывает содержимое выходного значения
     * фрагментного шейдера поверх старого значения. Однакоподдеживаются смешивание этих значнеий со значениями,
     * уже находящимися во фрейм буфере, и выполнение простых логических операций между выходными значениями фрагментного
     * шейдера и текущим содержанием фреймбуфера.*/

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment(6);
    for(uint32_t index=0;index<colorBlendAttachment.size();index++)
    {
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_FALSE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].alphaBlendOp = VK_BLEND_OP_MAX;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;                                                 //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
    colorBlending.logicOp = VK_LOGIC_OP_COPY;                                               //Optional
    colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachment.size());     //количество подключений
    colorBlending.pAttachments = colorBlendAttachment.data();                               //массив подключений
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional


    /* Для того чтобы сделать небольште изменения состояния более удобными, Vulkan предоставляет возможность помечать
     * определенные части графического конвейера как динамически, что значит что они могут быть изменены прямо на месте
     * при помощи команд прямо внутри командного буфера*/

    // добавлено

    std::array<VkDescriptorSetLayout,1> SetLayouts = {skyBox.descriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &skyBox.pipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;                                            //число структур в массиве структур
    pipelineInfo.pStages = shaderStages;                                    //указывает на массив структур VkPipelineShaderStageCreateInfo, каждая из которых описыват одну стадию
    pipelineInfo.pVertexInputState = &vertexInputInfo;                      //вершинный ввод
    pipelineInfo.pInputAssemblyState = &inputAssembly;                      //фаза входной сборки
    pipelineInfo.pViewportState = &viewportState;                           //Преобразование области вывода
    pipelineInfo.pRasterizationState = &rasterizer;                         //растеризация
    pipelineInfo.pMultisampleState = &multisampling;                        //мультсемплинг
    pipelineInfo.pColorBlendState = &colorBlending;                         //смешивание цветов
    pipelineInfo.layout = skyBox.pipelineLayout;                            //
    pipelineInfo.renderPass = renderPass;                                   //проход рендеринга
    pipelineInfo.subpass = 0;                                               //подпроход рендеригка
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &depthStencil;

    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &skyBox.pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

//===================secondPipline===================================

void graphics::createSecondDescriptorSetLayout()
{
    uint32_t index;
    std::vector<VkDescriptorSetLayoutBinding> Binding(9);
    for(index = 0; index<6;index++)
    {
        Binding.at(index).binding = index;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        Binding.at(index).descriptorCount = 1;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    }
        Binding.at(index).binding = index;
        Binding.at(index).descriptorCount = MAX_LIGHT_SOURCE_COUNT;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    index++;
        Binding.at(index).binding = index;
        Binding.at(index).descriptorCount = MAX_LIGHT_SOURCE_COUNT;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    index++;
        Binding.at(index).binding = index;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        Binding.at(index).descriptorCount = 1;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(Binding.size());
        layoutInfo.pBindings = Binding.data();

    if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &secondDescriptorSetLayout) != VK_SUCCESS)
    {throw std::runtime_error("failed to create descriptor set layout!");}
}

void graphics::createSecondDescriptorPool()
{
    size_t index = 0;
    std::vector<VkDescriptorPoolSize> poolSizes(6+2*MAX_LIGHT_SOURCE_COUNT+1);
    for(index = 0; index<6;index++)
    {
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    }
    for(size_t i=0;i<MAX_LIGHT_SOURCE_COUNT;i++)
    {
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
        index++;
    }
    for(size_t i=0;i<MAX_LIGHT_SOURCE_COUNT;i++)
    {
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
        index++;
    }
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;                           //Сначала нам нужно описать, какие типы дескрипторов будут содержать наши наборы дескрипторов
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);                //и сколько их, используя VkDescriptorPoolSizeструктуры.

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(imageCount);

    if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &secondDescriptorPool) != VK_SUCCESS)
    {throw std::runtime_error("failed to create descriptor pool!");}
}

void graphics::createSecondDescriptorSets(const std::vector<light<spotLight>*> & lightSource)
{
    std::vector<VkDescriptorSetLayout> layouts(imageCount, secondDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = secondDescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();

    secondDescriptorSets.resize(imageCount);
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, secondDescriptorSets.data()) != VK_SUCCESS)
    {throw std::runtime_error("failed to allocate descriptor sets!");}

    //Наборы дескрипторов уже выделены, но дескрипторы внутри еще нуждаются в настройке.
    //Теперь мы добавим цикл для заполнения каждого дескриптора:
    for (size_t i = 0; i < imageCount; i++)
    {
        uint32_t index = 0;
        std::vector<VkWriteDescriptorSet> descriptorWrites(9);
        std::vector<VkDescriptorImageInfo> imageInfo(6);
        for(index = 0; index<6;index++)
        {
            imageInfo.at(index).imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.at(index).imageView = Attachments.at(3+index).imageView.at(i);
            imageInfo.at(index).sampler = Attachments.at(3+index).sampler;

            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = secondDescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pImageInfo = &imageInfo.at(index);
        }

        VkDescriptorBufferInfo lightBufferInfo[MAX_LIGHT_SOURCE_COUNT];
        for (size_t j = 0; j < lightSource.size(); j++)
        {
            lightBufferInfo[j].buffer = lightSource.at(j)->getLightUniformBuffers().at(i);
            lightBufferInfo[j].offset = 0;
            lightBufferInfo[j].range = sizeof(LightUniformBufferObject);
        }
        for (size_t j = lightSource.size(); j < MAX_LIGHT_SOURCE_COUNT; j++)
        {
            lightBufferInfo[j].buffer = emptyUniformBuffers[i];
            lightBufferInfo[j].offset = 0;
            lightBufferInfo[j].range = sizeof(LightUniformBufferObject);
        }

        VkDescriptorImageInfo shadowImageInfo[MAX_LIGHT_SOURCE_COUNT];
        for (size_t j = 0; j < lightSource.size(); j++)
        {
            shadowImageInfo[j].imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            shadowImageInfo[j].imageView    = lightSource.at(j)->getShadowEnable() ? lightSource.at(j)->getImageView() : emptyTexture->getTextureImageView();
            shadowImageInfo[j].sampler      = lightSource.at(j)->getShadowEnable() ? lightSource.at(j)->getSampler() : emptyTexture->getTextureSampler();
        }
        for (size_t j = lightSource.size(); j < MAX_LIGHT_SOURCE_COUNT; j++)
        {
            shadowImageInfo[j].imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            shadowImageInfo[j].imageView    = emptyTexture->getTextureImageView();
            shadowImageInfo[j].sampler      = emptyTexture->getTextureSampler();
        }

        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = secondDescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.at(index).descriptorCount = MAX_LIGHT_SOURCE_COUNT;
            descriptorWrites.at(index).pBufferInfo = lightBufferInfo;
        index++;
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = secondDescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.at(index).descriptorCount = MAX_LIGHT_SOURCE_COUNT;
            descriptorWrites.at(index).pImageInfo = shadowImageInfo;
        index++;
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = secondDescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void graphics::createSecondPipelines()
{
    //считываем шейдеры
    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\secondPass\\secondvert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\secondPass\\secondfrag.spv");
    //создаём шейдерные модули
    VkShaderModule vertShaderModule = createShaderModule(app, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(app, fragShaderCode);
    //задаём стадии шейдеров в конвейере
    //вершинный
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;                             //ниформацию о всех битах смотри на странице 222
    vertShaderStageInfo.module = vertShaderModule;                                      //сюда передаём шейдерный модуль
    vertShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа
    //фрагментный
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;                           //ниформацию о всех битах смотри на странице 222
    fragShaderStageInfo.module = fragShaderModule;                                      //сюда передаём шейдерный модуль
    fragShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа
    //формаируем нужный массив, который будем передавать в структуру для создания графического конвейера
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    /* Для рендеринга настоящей геометрии вам необходимо передавать данные в конвайер Vulkan.
     * Вы можете использовать индексы вершин и экземпляров, доступные в SPIR-V, для автоматической
     * генерации геометрии или же явно извлекать геометрические данные из буфера. Вместо этого вы можете
     * описать размещение геометрических данных в памяти, и Vulkan может сам извлекать эти данные для вас, передавая их прямо в шейдер*/

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;                                                      //количество привязанных дескрипторов вершин
    vertexInputInfo.vertexAttributeDescriptionCount = 0;  //количество дескрипторов атрибутов вершин
    vertexInputInfo.pVertexBindingDescriptions = nullptr;                                       //указатель на массив соответствующийх структуру
    vertexInputInfo.pVertexAttributeDescriptions = nullptr;                            //указатель на массив соответствующийх структуру

    /* фаза входной сборки графического конвейера берёт данные в вершинах и группирует их в примитивы,
     * готовые для обработки следубщими стадиями конвейера.*/

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;                       //тип примитива, подробно про тип примитива смотри со страницы 228
    inputAssembly.primitiveRestartEnable = VK_FALSE;                                    //это флаг, кторый используется для того, чтоюы можно было оборвать примитивы полосы и веера и затем начать их снова
                                                                                        //без него кажда полоса и веер потребуют отдельной команды вывода.
    /* Преобразование области вывода - это последнее преобразование координат в конвейере Vulkan до растретизации.
     * Оно преобразует координаты вершины из нормализованных координат устройства в оконные координаты. Одновременно
     * может использоваться несколько областей вывода.*/

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) swapChainExtent.width;
    viewport.height = (float) swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;                                                //число областей вывода
    viewportState.pViewports = &viewport;                                           //размер каждой области вывода
    viewportState.scissorCount = 1;                                                 //число прямоугольников
    viewportState.pScissors = &scissor;                                             //эксцент

    /* Растеризация - это процесс, в ходе которого примитивы, представленные вершинами, преобразуются в потоки фрагментов, которых к обработке
     * фрагментным шейдером. Состояние растеризации управляется тем, как этот процесс происходит, и задаётся при помощи следующей структуры*/

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;                                      //используется для того чтобы полностью выключить растеризацию. Когда флаг установлен, растеризация не работает и не создаются фрагменты
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;                                      //используется для того чтобы Vulkan автоматически превращал треугольники в точки или отрезки
    rasterizer.lineWidth = 1.0f;                                                        //толщина линии
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;                                        //параметр обрасывания
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;                             //параметр направления обхода (против часовой стрелки)
    rasterizer.depthBiasEnable = VK_FALSE;                                              //используется для того чтобы включать отсечение глубины
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional                              //
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    /* Мультсемплинг - это процесс создания нескольких образцов (sample) для каждого пиксела в изображении.
     * Они используются для борьбы с алиансингом и может заметно улучшить общее качество изображения при эффективном использовании*/

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = msaaSamples;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional

    /* Последней стадией в графическом конвейере является стадия смешивания цветов. Эта стадия отвечает за запись фрагментов
     * в цветовые подключения. Во многих случаях это простая операция, которая просто записывает содержимое выходного значения
     * фрагментного шейдера поверх старого значения. Однакоподдеживаются смешивание этих значнеий со значениями,
     * уже находящимися во фрейм буфере, и выполнение простых логических операций между выходными значениями фрагментного
     * шейдера и текущим содержанием фреймбуфера.*/

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment(3);
    for(uint32_t index=0;index<colorBlendAttachment.size();index++)
    {
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_FALSE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].alphaBlendOp = VK_BLEND_OP_MAX;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;                                                 //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
    colorBlending.logicOp = VK_LOGIC_OP_COPY;                                               //Optional
    colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachment.size());     //количество подключений
    colorBlending.pAttachments = colorBlendAttachment.data();                               //массив подключений
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional


    /* Для того чтобы сделать небольште изменения состояния более удобными, Vulkan предоставляет возможность помечать
     * определенные части графического конвейера как динамически, что значит что они могут быть изменены прямо на месте
     * при помощи команд прямо внутри командного буфера*/

    // добавлено

    std::array<VkDescriptorSetLayout,1> SetLayouts = {secondDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &secondPipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;                                            //число структур в массиве структур
    pipelineInfo.pStages = shaderStages;                                    //указывает на массив структур VkPipelineShaderStageCreateInfo, каждая из которых описыват одну стадию
    pipelineInfo.pVertexInputState = &vertexInputInfo;                      //вершинный ввод
    pipelineInfo.pInputAssemblyState = &inputAssembly;                      //фаза входной сборки
    pipelineInfo.pViewportState = &viewportState;                           //Преобразование области вывода
    pipelineInfo.pRasterizationState = &rasterizer;                         //растеризация
    pipelineInfo.pMultisampleState = &multisampling;                        //мультсемплинг
    pipelineInfo.pColorBlendState = &colorBlending;                         //смешивание цветов
    pipelineInfo.layout = secondPipelineLayout;
    pipelineInfo.renderPass = renderPass;                                   //проход рендеринга
    pipelineInfo.subpass = 1;                                               //подпроход рендеригка
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &depthStencil;

    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &secondGraphicsPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
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

        swapChainFramebuffers.resize(imageCount);
        for (size_t image = 0; image < imageCount; image++)
        {
            std::vector<VkImageView> attachments;

            for(size_t i=0;i<Attachments.size();i++)
            {
                attachments.push_back(Attachments[i].imageView[image]);
            }
            attachments.push_back(depthAttachment.imageView);

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;                                                                            //дескриптор объекта прохода рендеринга
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());                                        //число изображений
            framebufferInfo.pAttachments = attachments.data();                                                                  //набор изображений, которые должны быть привязаны к фреймбуферу, передаётся через массив дескрипторов объектов VkImageView
            framebufferInfo.width = swapChainExtent.width;                                                                      //ширина изображения
            framebufferInfo.height = swapChainExtent.height;                                                                    //высота изображения
            framebufferInfo.layers = 1;                                                                                         //число слоёв

            if (vkCreateFramebuffer(app->getDevice(), &framebufferInfo, nullptr, &swapChainFramebuffers[image]) != VK_SUCCESS)  //создание буфера кадров
            {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }
    void graphics::multiSampleFrameBuffer()
    {
        /* Фреймбуфер (буфер кадра) - эо объект, представляющий набор изображений, в который
         * графические конвейеры будут осуществлять рендеринг. Они затрагивают посление несколько
         * стадий в кнвейере: тесты глубины и трафарета, смешивание цветов, логические операции,
         * мультисемплинг и т.п. Фреймбуфер создаётся, используя ссылку на проход рендеринга, и может быть
         * использован с любым проходом рендеринга, имеющим похожую структуру подключений.*/

        swapChainFramebuffers.resize(imageCount);
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
            framebufferInfo.width = swapChainExtent.width;                                                                      //ширина изображения
            framebufferInfo.height = swapChainExtent.height;                                                                    //высота изображения
            framebufferInfo.layers = 1;                                                                                         //число слоёв

            if (vkCreateFramebuffer(app->getDevice(), &framebufferInfo, nullptr, &swapChainFramebuffers[image]) != VK_SUCCESS)  //создание буфера кадров
            {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }


void graphics::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i, std::vector<object*> & object3D, object & sky)
{
    std::array<VkClearValue, 10> clearValues{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[2].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[3].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[4].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[5].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[6].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[7].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[8].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[9].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo drawRenderPassInfo{};
    drawRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    drawRenderPassInfo.renderPass = renderPass;
    drawRenderPassInfo.framebuffer = swapChainFramebuffers[i];
    drawRenderPassInfo.renderArea.offset = {0, 0};
    drawRenderPassInfo.renderArea.extent = swapChainExtent;
    drawRenderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    drawRenderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[i], &drawRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        for(size_t j = 0; j<object3D.size() ;j++)
        {
            VkDeviceSize offsets[1] = { 0 };

            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, *object3D[j]->getPipeline());

            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, & object3D[j]->getModel()->vertices.buffer, offsets);
            if (object3D[j]->getModel()->indices.buffer != VK_NULL_HANDLE)
            {
                vkCmdBindIndexBuffer(commandBuffers[i],  object3D[j]->getModel()->indices.buffer, 0, VK_INDEX_TYPE_UINT32);
            }

            for (auto node : object3D[j]->getModel()->nodes)
            {
//                glm::vec3 position = glm::vec3(object3D[j]->getTransformation()*node->matrix*glm::vec4(0.0f,0.0f,0.0f,1.0f));
//                if(glm::length(position-cameraPosition)<object3D[j]->getVisibilityDistance()){
//                    renderNode(node,commandBuffers[i],descriptorSets[i],object3D[j]->getDescriptorSet()[i],*object3D[j]->getPipelineLayout());
//                }

                renderNode(node,commandBuffers[i],descriptorSets[i],object3D[j]->getDescriptorSet()[i],*object3D[j]->getPipelineLayout());
            }
        }

        VkDeviceSize offsets[1] = { 0 };
        vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, & sky.getModel()->vertices.buffer, offsets);
        vkCmdBindIndexBuffer(commandBuffers[i],  sky.getModel()->indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, skyBox.pipeline);
        vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, skyBox.pipelineLayout, 0, 1, &skyBox.descriptorSets[i], 0, NULL);

        vkCmdDrawIndexed(commandBuffers[i], 36, 1, 0, 0, 0);

    vkCmdNextSubpass(commandBuffers[i], VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, secondGraphicsPipeline);
        vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, secondPipelineLayout, 0, 1, &secondDescriptorSets[i], 0, nullptr);
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

            vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstBlockMaterial), &pushConstBlockMaterial);

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
    {
        renderNode(child, commandBuffer, descriptorSet,objectDescriptorSet,layout);
    }
}

void graphics::updateUniformBuffer(uint32_t currentImage, camera *cam, object *skybox)
{
    UniformBufferObject ubo{};

    ubo.view = cam->getViewMatrix();
    ubo.proj = glm::perspective(glm::radians(45.0f), (float) swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 1000.0f);
    ubo.proj[1][1] *= -1;
    ubo.eyePosition = glm::vec4(cam->getTranslate(), 1.0);

    cameraPosition = glm::vec4(cam->getTranslate(), 1.0);

    void* data;
    vkMapMemory(app->getDevice(), uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(app->getDevice(), uniformBuffersMemory[currentImage]);

    SkyboxUniformBufferObject SkyboxUbo{};

    SkyboxUbo.view = cam->getViewMatrix();
    SkyboxUbo.proj = glm::perspective(glm::radians(45.0f), (float) swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 1000.0f);
    SkyboxUbo.proj[1][1] *= -1;
    SkyboxUbo.model = glm::translate(glm::mat4x4(1.0f),cam->getTranslate())*skybox->getTransformation();

    vkMapMemory(app->getDevice(), skyBox.uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &SkyboxUbo, sizeof(SkyboxUbo));
    vkUnmapMemory(app->getDevice(), skyBox.uniformBuffersMemory[currentImage]);
}

uint32_t                        &graphics::ImageCount(){return imageCount;}

VkPipeline                      &graphics::PipeLine(){return graphicsPipeline;}
VkPipeline                      &graphics::BloomSpriteGraphicsPipeline(){return bloomSpriteGraphicsPipeline;}
VkPipeline                      &graphics::GodRaysPipeline(){return godRaysPipeline;}
VkPipeline                      &graphics::SkyBoxPipeLine(){return skyBox.pipeline;}

VkPipelineLayout                &graphics::PipelineLayout(){return pipelineLayout;}
VkPipelineLayout                &graphics::BloomSpritePipelineLayout(){return bloomSpritePipelineLayout;}
VkPipelineLayout                &graphics::GodRaysPipelineLayout(){return godRaysPipelineLayout;}
VkPipelineLayout                &graphics::SkyBoxPipelineLayout(){return skyBox.pipelineLayout;}

VkDescriptorSetLayout           &graphics::DescriptorSetLayout(){return descriptorSetLayout;}

std::vector<attachments>        &graphics::getAttachments(){return Attachments;}

VkFormat                        &graphics::SwapChainImageFormat(){return swapChainImageFormat;}

VkExtent2D                      &graphics::SwapChainExtent(){return swapChainExtent;}

VkRenderPass                    &graphics::RenderPass(){return renderPass;}

std::vector<VkFramebuffer>      &graphics::SwapChainFramebuffers(){return swapChainFramebuffers;}

VkDescriptorPool                &graphics::DescriptorPool(){return descriptorPool;}

std::vector<VkBuffer>           &graphics::UniformBuffers(){return uniformBuffers;}

std::vector<VkDeviceMemory>     &graphics::UniformBuffersMemory(){return uniformBuffersMemory;}

std::vector<VkDescriptorSet>    &graphics::DescriptorSets(){return descriptorSets;}
