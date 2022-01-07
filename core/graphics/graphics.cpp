#include "graphics.h"
#include "core/operations.h"
#include "core/transformational/object.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/light.h"
#include "core/transformational/gltfmodel.h"

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

    vkDestroyRenderPass(app->getDevice(), renderPass, nullptr);

    for(size_t i=0;i<Attachments.size();i++)
    {
        Attachments.at(i).deleteAttachment(&app->getDevice());
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

    for (size_t i = 0; i < imageCount; i++)
    {
        vkDestroyBuffer(app->getDevice(), uniformBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), uniformBuffersMemory[i], nullptr);

        vkDestroyBuffer(app->getDevice(), emptyUniformBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), emptyUniformBuffersMemory[i], nullptr);
    }
}

void graphics::setImageProp(uint32_t imageCount, VkFormat format, VkExtent2D extent)
{
    this->imageCount = imageCount;
    swapChainImageFormat = format;
    swapChainExtent = extent;
}

//=========================================================================//

void graphics::createColorAttachments()
{
    colorAttachments.resize(3);

    for(size_t i=0;i<colorAttachments.size();i++)
    {
        createImage(app,swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, swapChainImageFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorAttachments.at(i).image, colorAttachments.at(i).imageMemory);
        colorAttachments.at(i).imageView = createImageView(app, colorAttachments.at(i).image, swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
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
    Attachments.resize(3);
    for(size_t i=0;i<3;i++)
    {
        Attachments[i].resize(imageCount);
        for(size_t image=0; image<imageCount; image++)
        {
            createImage(app,swapChainExtent.width,swapChainExtent.height,1,VK_SAMPLE_COUNT_1_BIT,swapChainImageFormat,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Attachments[i].image[image], Attachments[i].imageMemory[image]);
            Attachments[i].imageView[image] = createImageView(app, Attachments[i].image[image], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
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

//=======================================RenderPass======================//

void graphics::createDrawRenderPass()
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

    for(size_t i=0;i<colorAttachments.size();i++)
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;                              //это поле задаёт формат подключений. Должно соответствовать фомрату используемого изображения
        colorAttachment.samples = msaaSamples;                            //задаёт число образцов в изображении и используется при мультисемплинге. VK_SAMPLE_COUNT_1_BIT - означает что мультисемплинг не используется
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                       //следующие 4 параметра смотри на странице 210
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                  //в каком размещении будет изображение в начале прохода
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;              //в каком размещении его нужно оставить по завершению рендеринга
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

    for(size_t i=0;i<Attachments.size();i++)
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

    VkAttachmentReference attachmentRef[colorAttachments.size()];
    for (size_t i=0;i<colorAttachments.size();i++)
    {
        attachmentRef[i].attachment = i;                                          //индекс в массив подключений
        attachmentRef[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;       //размещение
    }

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = colorAttachments.size();
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference resolveRef[Attachments.size()];
    for (size_t i=0;i<Attachments.size();i++)
    {
        resolveRef[i].attachment = colorAttachments.size()+1+i;
        resolveRef[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    VkSubpassDescription subpass{};                                                 //подпроходы рендеринга
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;                    //бит для графики
    subpass.colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size());  //количество подключений
    subpass.pColorAttachments = attachmentRef;                                      //подключения
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = resolveRef;

    VkSubpassDependency dependency{};                                           //зависимости
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;                                //ссылка из исходного прохода (создавшего данные)
    dependency.dstSubpass = 0;                                                  //в целевой подпроход (поглощающий данные)
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;       //задаёт как стадии конвейера в исходном проходе создают данные
    dependency.srcAccessMask = 0;                                                                                               //поля задают как каждый из исходных проходов обращается к данным
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());         //количество структур VkAtachmentDescription, определяющих подключения, связанные с этим проходом рендеринга
    renderPassInfo.pAttachments = attachments.data();                                   //Каждая структура определяет одно изображение, которое будет использовано как входное, выходное или входное и выходное одновремнно для оного или нескольких проходо в данном редеринге
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(app->getDevice(), &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)    //создаём проход рендеринга
    {
        throw std::runtime_error("failed to create render pass!");
    }
}

//==============================Piplines=================================

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

    VkDescriptorSetLayoutBinding lightUboLayoutBinding={};
    lightUboLayoutBinding.binding = 1;
    lightUboLayoutBinding.descriptorCount = MAX_LIGHT_SOURCE_COUNT;
    lightUboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    lightUboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    lightUboLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding shadowLayoutBinding{};
    shadowLayoutBinding.binding = 2;
    shadowLayoutBinding.descriptorCount = MAX_LIGHT_SOURCE_COUNT;
    shadowLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    shadowLayoutBinding.pImmutableSamplers = nullptr;

    std::array<VkDescriptorSetLayoutBinding, 3> bindings = {uboLayoutBinding,lightUboLayoutBinding,shadowLayoutBinding};
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
        std::vector<VkDescriptorPoolSize> poolSizes(1+MAX_LIGHT_SOURCE_COUNT+MAX_LIGHT_SOURCE_COUNT);
        size_t index = 0;

        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;                           //Сначала нам нужно описать, какие типы дескрипторов будут содержать наши наборы дескрипторов
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);                //и сколько их, используя VkDescriptorPoolSizeструктуры.
        index++;

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

void graphics::createDescriptorSets(const std::vector<light<spotLight>*> & lightSource, const std::vector<object*> & object3D)
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
                shadowImageInfo[j].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                shadowImageInfo[j].imageView = lightSource.at(j)->getImageView();
                shadowImageInfo[j].sampler = lightSource.at(j)->getSampler();
            }
            for (size_t j = lightSource.size(); j < MAX_LIGHT_SOURCE_COUNT; j++)
            {
                shadowImageInfo[j].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                shadowImageInfo[j].imageView = emptyTexture->getTextureImageView();
                shadowImageInfo[j].sampler = emptyTexture->getTextureSampler();
            }

            std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[1].descriptorCount = MAX_LIGHT_SOURCE_COUNT;
            descriptorWrites[1].pBufferInfo = lightBufferInfo;

            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = descriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[2].descriptorCount = MAX_LIGHT_SOURCE_COUNT;
            descriptorWrites[2].pImageInfo = shadowImageInfo;

            vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }
}

void graphics::createGraphicsPipeline()
{
    //считываем шейдеры
    auto vertShaderCode = readFile("C:\\Users\\kiril\\OneDrive\\qt\\vulkan\\core\\graphics\\shaders\\base\\basevert.spv");
    auto fragShaderCode = readFile("C:\\Users\\kiril\\OneDrive\\qt\\vulkan\\core\\graphics\\shaders\\base\\basefrag.spv");
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

    VkPipelineColorBlendAttachmentState colorBlendAttachment[3];
    colorBlendAttachment[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment[0].blendEnable = VK_FALSE;
    colorBlendAttachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[0].colorBlendOp = VK_BLEND_OP_MAX;
    colorBlendAttachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[0].alphaBlendOp = VK_BLEND_OP_MAX;

    colorBlendAttachment[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment[1].blendEnable = VK_TRUE;
    colorBlendAttachment[1].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[1].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[1].colorBlendOp = VK_BLEND_OP_MAX;
    colorBlendAttachment[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[1].alphaBlendOp = VK_BLEND_OP_MAX;

    colorBlendAttachment[2].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment[2].blendEnable = VK_FALSE;
    colorBlendAttachment[2].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[2].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[2].colorBlendOp = VK_BLEND_OP_MAX;
    colorBlendAttachment[2].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[2].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[2].alphaBlendOp = VK_BLEND_OP_MAX;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;                                         //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
    colorBlending.logicOp = VK_LOGIC_OP_COPY;                                       //Optional
    colorBlending.attachmentCount = 3;                                              //количество подключений
    colorBlending.pAttachments = colorBlendAttachment;                              //массив подключений
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
    auto vertShaderCode = readFile("C:\\Users\\kiril\\OneDrive\\qt\\vulkan\\core\\graphics\\shaders\\bloomSprite\\vertBloomSprite.spv");
    auto fragShaderCode = readFile("C:\\Users\\kiril\\OneDrive\\qt\\vulkan\\core\\graphics\\shaders\\bloomSprite\\fragBloomSprite.spv");
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

    VkPipelineColorBlendAttachmentState colorBlendAttachment[3];
    colorBlendAttachment[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment[0].blendEnable = VK_FALSE;
    colorBlendAttachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[0].colorBlendOp = VK_BLEND_OP_MAX;
    colorBlendAttachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[0].alphaBlendOp = VK_BLEND_OP_MAX;

    colorBlendAttachment[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment[1].blendEnable = VK_FALSE;
    colorBlendAttachment[1].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[1].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[1].colorBlendOp = VK_BLEND_OP_MAX;
    colorBlendAttachment[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[1].alphaBlendOp = VK_BLEND_OP_MAX;

    colorBlendAttachment[2].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment[2].blendEnable = VK_FALSE;
    colorBlendAttachment[2].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[2].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[2].colorBlendOp = VK_BLEND_OP_MAX;
    colorBlendAttachment[2].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[2].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[2].alphaBlendOp = VK_BLEND_OP_MAX;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;                                         //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
    colorBlending.logicOp = VK_LOGIC_OP_COPY;                                       //Optional
    colorBlending.attachmentCount = 3;                                              //количество подключений
    colorBlending.pAttachments = colorBlendAttachment;                              //массив подключений
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
    depthStencil.depthWriteEnable = VK_FALSE;
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
    pipelineInfo.layout = bloomSpritePipelineLayout;                                   //
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
    auto vertShaderCode = readFile("C:\\Users\\kiril\\OneDrive\\qt\\vulkan\\core\\graphics\\shaders\\godRays\\godRaysVert.spv");
    auto fragShaderCode = readFile("C:\\Users\\kiril\\OneDrive\\qt\\vulkan\\core\\graphics\\shaders\\godRays\\godRaysFrag.spv");
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

    VkPipelineColorBlendAttachmentState colorBlendAttachment[3];
    colorBlendAttachment[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment[0].blendEnable = VK_FALSE;
    colorBlendAttachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[0].colorBlendOp = VK_BLEND_OP_MAX;
    colorBlendAttachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[0].alphaBlendOp = VK_BLEND_OP_MAX;

    colorBlendAttachment[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment[1].blendEnable = VK_FALSE;
    colorBlendAttachment[1].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[1].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[1].colorBlendOp = VK_BLEND_OP_MAX;
    colorBlendAttachment[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[1].alphaBlendOp = VK_BLEND_OP_MAX;

    colorBlendAttachment[2].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment[2].blendEnable = VK_FALSE;
    colorBlendAttachment[2].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[2].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[2].colorBlendOp = VK_BLEND_OP_MAX;
    colorBlendAttachment[2].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment[2].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment[2].alphaBlendOp = VK_BLEND_OP_MAX;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;                                         //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
    colorBlending.logicOp = VK_LOGIC_OP_COPY;                                       //Optional
    colorBlending.attachmentCount = 3;                                              //количество подключений
    colorBlending.pAttachments = colorBlendAttachment;                              //массив подключений
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

//===================Framebuffers===================================

void graphics::createFramebuffers()
{
    /* Фреймбуфер (буфер кадра) - эо объект, представляющий набор изображений, в который
     * графические конвейеры будут осуществлять рендеринг. Они затрагивают посление несколько
     * стадий в кнвейере: тесты глубины и трафарета, смешивание цветов, логические операции,
     * мультисемплинг и т.п. Фреймбуфер создаётся, используя ссылку на проход рендеринга, и может быть
     * использован с любым проходом рендеринга, имеющим похожую структуру подключений.*/

    swapChainFramebuffers.resize(imageCount);
    for (size_t image = 0; image < imageCount; image++)
    {
        std::vector<VkImageView> attachments(colorAttachments.size()+Attachments.size()+1);
        for(size_t i=0;i<colorAttachments.size();i++)
        {
            attachments[i] = colorAttachments[i].imageView;
        }
        attachments[colorAttachments.size()] = depthAttachment.imageView;
        for(size_t i=0;i<Attachments.size();i++)
        {
            attachments[colorAttachments.size()+1+i] = Attachments[i].imageView[image];
        }

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;                                                                    //дескриптор объекта прохода рендеринга
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());                                //число изображений
        framebufferInfo.pAttachments = attachments.data();                                                          //набор изображений, которые должны быть привязаны к фреймбуферу, передаётся через массив дескрипторов объектов VkImageView
        framebufferInfo.width = swapChainExtent.width;                                                              //ширина изображения
        framebufferInfo.height = swapChainExtent.height;                                                            //высота изображения
        framebufferInfo.layers = 1;                                                                                 //число слоёв

        if (vkCreateFramebuffer(app->getDevice(), &framebufferInfo, nullptr, &swapChainFramebuffers[image]) != VK_SUCCESS) //создание буфера кадров
        {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void graphics::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i, std::vector<object*> & object3D)
{
    std::array<VkClearValue, 4> clearValues{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[2].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[3].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo drawRenderPassInfo{};
    drawRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    drawRenderPassInfo.renderPass = renderPass;
    drawRenderPassInfo.framebuffer = swapChainFramebuffers[i];
    drawRenderPassInfo.renderArea.offset = {0, 0};
    drawRenderPassInfo.renderArea.extent = swapChainExtent;
    drawRenderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    drawRenderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[i], &drawRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);


//        for(size_t j = 0; j<object3D.size() ;j++)
//        {
//            if(object3D[j]->isEnableBloomSprite())
//            {
//                VkDeviceSize offsets[] = {0};
//                vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, bloomSpriteGraphicsPipeline);
//                vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, &object3D[j]->getVertexBuffer(), offsets);

//                VkDescriptorSet descriptors[3] = {descriptorSets[i],object3D[j]->getTextureDescriptorSet()[i],object3D[j]->getNormalMapDescriptorSet()[i]};
//                vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, bloomSpritePipelineLayout, 0, 3, descriptors, 0, nullptr);

//                PushConstant pc;
//                pc.modelMatrix = object3D[j]->getTransformation();

//                vkCmdPushConstants(commandBuffers[i], *object3D[j]->getPipelineLayout(), VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant), (void *)&pc);
//                vkCmdDraw(commandBuffers[i], 4, 1, 0, 0);
//            }
//        }

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
                renderNode(node,commandBuffers[i],descriptorSets[i],object3D[j]->getDescriptorSet()[i],*object3D[j]->getPipelineLayout());
            }
        }

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

uint32_t                        &graphics::ImageCount(){return imageCount;}

VkPipeline                      &graphics::PipeLine(){return graphicsPipeline;}
VkPipeline                      &graphics::BloomSpriteGraphicsPipeline(){return bloomSpriteGraphicsPipeline;}
VkPipeline                      &graphics::GodRaysPipeline(){return godRaysPipeline;}

VkPipelineLayout                &graphics::PipelineLayout(){return pipelineLayout;}
VkPipelineLayout                &graphics::BloomSpritePipelineLayout(){return bloomSpritePipelineLayout;}
VkPipelineLayout                &graphics::GodRaysPipelineLayout(){return godRaysPipelineLayout;}

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
