#include "shadowGraphics.h"
#include "core/transformational/object.h"
#include "core/operations.h"
#include "core/transformational/gltfmodel.h"

shadowGraphics::shadowGraphics(VkApplication *app, uint32_t imageCount): app(app)
{
    image.Count = imageCount;
    image.Extent.width = 1024;
    image.Extent.height = 1024;
}

void shadowGraphics::createMap()
{
    VkFormat shadowFormat = findDepthFormat(app);
    createImage(app,image.Extent.width,image.Extent.height,image.MipLevels,VK_SAMPLE_COUNT_1_BIT,shadowFormat,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthAttachment.image, depthAttachment.imageMemory);
}

void shadowGraphics::createMapView()
{
    VkFormat shadowFormat = findDepthFormat(app);
    depthAttachment.imageView = createImageView(app,depthAttachment.image, shadowFormat, VK_IMAGE_ASPECT_DEPTH_BIT, image.MipLevels);
}

void shadowGraphics::destroy()
{
    depthAttachment.deleteAttachment(&app->getDevice());

    shadow.Destroy(app);

    for(uint32_t i=0;i<shadowMapFramebuffer.size();i++)
        vkDestroyFramebuffer(app->getDevice(), shadowMapFramebuffer.at(i),nullptr);
    vkDestroyRenderPass(app->getDevice(), RenderPass, nullptr);
    for(uint32_t i=0;i<shadowCommandPool.size();i++)
        vkFreeCommandBuffers(app->getDevice(), shadowCommandPool.at(i), static_cast<uint32_t>(shadowCommandBuffer.at(i).size()), shadowCommandBuffer.at(i).data());
    for(size_t i = 0; i < shadowCommandPool.size(); i++)
        vkDestroyCommandPool(app->getDevice(), shadowCommandPool.at(i), nullptr);
    vkDestroySampler(app->getDevice(), shadowSampler, nullptr);
}

void shadowGraphics::createSampler()
{
    float mipLevel = 1.0f;

    VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;                           //поля определяют как интерполировать тексели, которые увеличенные
        samplerInfo.minFilter = VK_FILTER_LINEAR;                           //или минимизированы
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;          //Режим адресации
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;          //Обратите внимание, что оси называются U, V и W вместо X, Y и Z. Это соглашение для координат пространства текстуры.
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;          //Повторение текстуры при выходе за пределы размеров изображения.
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;                                   //Чтобы выяснить, какое значение мы можем использовать, нам нужно получить свойства физического устройства
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;         //В этом borderColor поле указывается, какой цвет возвращается при выборке за пределами изображения в режиме адресации с ограничением по границе.
        samplerInfo.unnormalizedCoordinates = VK_FALSE;                     //поле определяет , какая система координат вы хотите использовать для адреса текселей в изображении
        samplerInfo.compareEnable = VK_FALSE;                               //Если функция сравнения включена, то тексели сначала будут сравниваться со значением,
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;                       //и результат этого сравнения используется в операциях фильтрации
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.minLod = static_cast<float>(mipLevel*image.MipLevels);
        samplerInfo.maxLod = static_cast<float>(image.MipLevels);
        samplerInfo.mipLodBias = 0.0f;
    if (vkCreateSampler(app->getDevice(), &samplerInfo, nullptr, &shadowSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create texture sampler!");
}

void shadowGraphics::createRenderPass()
{
    VkAttachmentDescription attachments{};
        attachments.format =  findDepthFormat(app);
        attachments.samples = VK_SAMPLE_COUNT_1_BIT;
        attachments.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments.flags = 0;

    VkAttachmentReference depthRef{};
        depthRef.attachment = 0;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass;
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.flags = 0;
        subpass.inputAttachmentCount = 0;
        subpass.pInputAttachments = NULL;
        subpass.colorAttachmentCount = 0;
        subpass.pColorAttachments = NULL;
        subpass.pResolveAttachments = NULL;
        subpass.pDepthStencilAttachment = &depthRef;
        subpass.preserveAttachmentCount = 0;
        subpass.pPreserveAttachments = NULL;

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.pNext = NULL;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &attachments;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 0;
        renderPassInfo.pDependencies = NULL;
        renderPassInfo.flags = 0;

    vkCreateRenderPass(app->getDevice(), &renderPassInfo, NULL, &RenderPass);
}

void shadowGraphics::createFramebuffer()
{
    shadowMapFramebuffer.resize(image.Count);
    for (size_t i = 0; i < shadowMapFramebuffer.size(); i++)
    {
        VkFramebufferCreateInfo framebufferInfo;
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.pNext = NULL;
            framebufferInfo.renderPass = RenderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &depthAttachment.imageView;
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
            framebufferInfo.flags = 0;
        vkCreateFramebuffer(app->getDevice(), &framebufferInfo, NULL, &shadowMapFramebuffer.at(i));
    }
}

void shadowGraphics::createCommandPool(uint32_t commandPoolsCount)
{
    shadowCommandPool.resize(commandPoolsCount);
    shadowCommandBuffer.resize(commandPoolsCount);
    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = app->getQueueFamilyIndices().graphicsFamily.value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    for(size_t i = 0; i < shadowCommandPool.size(); i++)
        if (vkCreateCommandPool(app->getDevice(), &poolInfo, nullptr, &shadowCommandPool.at(i)) != VK_SUCCESS)
            throw std::runtime_error("failed to create command pool!");
}

void shadowGraphics::Shadow::Destroy(VkApplication  *app)
{
    vkDestroyPipeline(app->getDevice(), Pipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), PipelineLayout,nullptr);
    vkDestroyDescriptorSetLayout(app->getDevice(), DescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(app->getDevice(), uniformBlockSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(app->getDevice(), uniformBufferSetLayout, nullptr);
    vkDestroyDescriptorPool(app->getDevice(), DescriptorPool, nullptr);
}

void shadowGraphics::Shadow::createPipeline(VkApplication *app, shadowInfo info)
{
    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\shadow\\shad.spv");
    VkShaderModule vertShaderModule = createShaderModule(app,vertShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;                             //ниформацию о всех битах смотри на странице 222
    vertShaderStageInfo.module = vertShaderModule;                                      //сюда передаём шейдерный модуль
    vertShaderStageInfo.pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo};

    auto bindingDescription = gltfModel::Vertex::getShadowBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getShadowAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;                                                      //количество привязанных дескрипторов вершин
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());  //количество дескрипторов атрибутов вершин
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;                                       //указатель на массив соответствующийх структуру
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();                            //указатель на массив соответствующийх структуру

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;                       //тип примитива, подробно про тип примитива смотри со страницы 228
    inputAssembly.primitiveRestartEnable = VK_FALSE;                                    //это флаг, кторый используется для того, чтоюы можно было оборвать примитивы полосы и веера и затем начать их снова
                                                                                        //без него кажда полоса и веер потребуют отдельной команды вывода.
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) info.width;
    viewport.height = (float) info.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent.width = info.width;
    scissor.extent.height = info.height;
    scissor.offset.x = 0;
    scissor.offset.y = 0;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;                                                //число областей вывода
    viewportState.pViewports = &viewport;                                           //размер каждой области вывода
    viewportState.scissorCount = 1;                                                 //число прямоугольников
    viewportState.pScissors = &scissor;                                             //эксцент

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;                                      //используется для того чтобы полностью выключить растеризацию. Когда флаг установлен, растеризация не работает и не создаются фрагменты
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;                                      //используется для того чтобы Vulkan автоматически превращал треугольники в точки или отрезки
    rasterizer.lineWidth = 1.0f;                                                        //толщина линии
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;                                        //параметр обрасывания
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;                             //параметр направления обхода (против часовой стрелки)
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 4.0f;
    rasterizer.depthBiasSlopeFactor = 1.5f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;                                         //задаёт, необходимо ли выполнить логические операции между выводом фрагментного шейдера и содержанием цветовых подключений
    colorBlending.logicOp = VK_LOGIC_OP_COPY;                                       //Optional
    colorBlending.attachmentCount = 1;                                              //количество подключений
    colorBlending.pAttachments = &colorBlendAttachment;                             //массив подключений

    VkPushConstantRange pushConstantRange;
        pushConstantRange.stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(uint32_t);
    std::array<VkDescriptorSetLayout,3> SetLayouts = {DescriptorSetLayout,uniformBufferSetLayout,uniformBufferSetLayout};
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

    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout!");

    VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 1;                                            //число структур в массиве структур
        pipelineInfo.pStages = shaderStages;                                    //указывает на массив структур VkPipelineShaderStageCreateInfo, каждая из которых описыват одну стадию
        pipelineInfo.pVertexInputState = &vertexInputInfo;                      //вершинный ввод
        pipelineInfo.pInputAssemblyState = &inputAssembly;                      //фаза входной сборки
        pipelineInfo.pViewportState = &viewportState;                           //Преобразование области вывода
        pipelineInfo.pRasterizationState = &rasterizer;                         //растеризация
        pipelineInfo.pMultisampleState = &multisampling;                        //мультсемплинг
        pipelineInfo.pColorBlendState = &colorBlending;                         //смешивание цветов
        pipelineInfo.layout = PipelineLayout;
        pipelineInfo.renderPass = info.renderPass;                              //проход рендеринга
        pipelineInfo.subpass = 0;                                               //подпроход рендеригка
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.pDepthStencilState = &depthStencil;
    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics pipeline!");

    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void shadowGraphics::Shadow::createDescriptorSetLayout(VkApplication *app)
{
    VkDescriptorSetLayoutBinding lightUboLayoutBinding={};
    lightUboLayoutBinding.binding = 1;
    lightUboLayoutBinding.descriptorCount = 1;
    lightUboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    lightUboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    lightUboLayoutBinding.pImmutableSamplers = nullptr;

    std::array<VkDescriptorSetLayoutBinding, 1> bindings = {lightUboLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &DescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor set layout!");

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
        throw std::runtime_error("failed to create descriptor set layout!");

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
        throw std::runtime_error("failed to create descriptor set layout!");
}

void shadowGraphics::createDescriptorPool()
{
    size_t index = 0;
    std::vector<VkDescriptorPoolSize> poolSizes(1);
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(image.Count);

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);

    if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &shadow.DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor pool!");
}

void shadowGraphics::createDescriptorSets(std::vector<VkBuffer> &lightUniformBuffers)
{
    std::vector<VkDescriptorSetLayout> layouts(image.Count, shadow.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = shadow.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();

    shadow.DescriptorSets.resize(image.Count);
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, shadow.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate descriptor sets!");

    for (size_t i = 0; i < image.Count; i++)
    {
        VkDescriptorBufferInfo lightBufferInfo;
            lightBufferInfo.buffer = lightUniformBuffers[i];
            lightBufferInfo.offset = 0;
            lightBufferInfo.range = sizeof(LightUniformBufferObject);
        VkWriteDescriptorSet descriptorWrites{};
            descriptorWrites.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.dstSet = shadow.DescriptorSets[i];
            descriptorWrites.dstBinding = 1;
            descriptorWrites.dstArrayElement = 0;
            descriptorWrites.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.descriptorCount = 1;
            descriptorWrites.pBufferInfo = &lightBufferInfo;
        vkUpdateDescriptorSets(app->getDevice(), 1, &descriptorWrites, 0, nullptr);
    }
}

void shadowGraphics::createCommandBuffers(uint32_t number)
{
    shadowCommandBuffer[number].resize(image.Count);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = shadowCommandPool[number];
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = image.Count;
    if (vkAllocateCommandBuffers(app->getDevice(), &allocInfo, shadowCommandBuffer[number].data()) != VK_SUCCESS)
    {throw std::runtime_error("failed to allocate command buffers!");}
}

void shadowGraphics::updateCommandBuffers(uint32_t number, uint32_t i, std::vector<object *> & object3D, uint32_t lightNumber)
{
    VkClearValue clearValues{};
        clearValues.depthStencil.depth = 1.0f;
        clearValues.depthStencil.stencil = 0;

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;                                            //поле для передачи информации о том, как будет использоваться этот командный буфер (смотри страницу 102)
        beginInfo.pInheritanceInfo = nullptr;                           //используется при начале вторичного буфера, для того чтобы определить, какие состояния наследуются от первичного командного буфера, который его вызовет
    if (vkBeginCommandBuffer(shadowCommandBuffer[number][i], &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording command buffer!");

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.pNext = NULL;
        renderPassInfo.renderPass = RenderPass;
        renderPassInfo.framebuffer = shadowMapFramebuffer[i];
        renderPassInfo.renderArea.offset.x = 0;
        renderPassInfo.renderArea.offset.y = 0;
        renderPassInfo.renderArea.extent.width = image.Extent.width;
        renderPassInfo.renderArea.extent.height = image.Extent.height;
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearValues;

    vkCmdBeginRenderPass(shadowCommandBuffer[number][i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport;
            viewport.width = image.Extent.width;
            viewport.height = image.Extent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            viewport.x = 0;
            viewport.y = 0;
        vkCmdSetViewport(shadowCommandBuffer[number][i], 0, 1, &viewport);

        VkRect2D scissor;
            scissor.extent.width = image.Extent.width;
            scissor.extent.height = image.Extent.height;
            scissor.offset.x = 0;
            scissor.offset.y = 0;
        vkCmdSetScissor(shadowCommandBuffer[number][i], 0, 1, &scissor);

        vkCmdBindPipeline(shadowCommandBuffer[number][i], VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.Pipeline);
        vkCmdPushConstants(shadowCommandBuffer[number][i], shadow.PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(uint32_t), &lightNumber);
        for(size_t j = 0; j<object3D.size() ;j++)
        {
            VkDeviceSize offsets[1] = { 0 };
            vkCmdBindVertexBuffers(shadowCommandBuffer[number][i], 0, 1, & object3D[j]->getModel()->vertices.buffer, offsets);

            if (object3D[j]->getModel()->indices.buffer != VK_NULL_HANDLE)
                vkCmdBindIndexBuffer(shadowCommandBuffer[number][i],  object3D[j]->getModel()->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

            for (auto node : object3D[j]->getModel()->nodes)
                renderNode(node,shadowCommandBuffer[number][i],shadow.DescriptorSets[i],object3D[j]->getDescriptorSet()[i]);

//            for (auto node : object3D[j]->getModel()->nodes)
//                if(glm::length(glm::vec3(object3D[j]->getTransformation()*node->matrix*glm::vec4(0.0f,0.0f,0.0f,1.0f))-camera->getTranslate())<object3D[j]->getVisibilityDistance()){
//                    renderNode(node,shadowCommandBuffer[number][i],shadow.DescriptorSets[i],object3D[j]->getDescriptorSet()[i]);
        }

    vkCmdEndRenderPass(shadowCommandBuffer[number][i]);

    if (vkEndCommandBuffer(shadowCommandBuffer[number][i]) != VK_SUCCESS)
        throw std::runtime_error("failed to record command buffer!");
}

void shadowGraphics::renderNode(Node *node, VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, VkDescriptorSet& objectDescriptorSet)
{
    if (node->mesh)
    {
        for (Primitive* primitive : node->mesh->primitives)
        {
            const std::vector<VkDescriptorSet> descriptorsets =
            {
                descriptorSet,
                objectDescriptorSet,
                node->mesh->uniformBuffer.descriptorSet
            };
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.PipelineLayout, 0, static_cast<uint32_t>(descriptorsets.size()), descriptorsets.data(), 0, NULL);
            if (primitive->hasIndices)
                vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
            else
                vkCmdDraw(commandBuffer, primitive->vertexCount, 1, 0, 0);
        }
    }
    for (auto child : node->children)
        renderNode(child, commandBuffer, descriptorSet,objectDescriptorSet);
}

void shadowGraphics::createShadow(uint32_t commandPoolsCount)
{
    createCommandPool(commandPoolsCount);
    createMap();
    createMapView();
    createSampler();
    createRenderPass();
    createFramebuffer();
    shadow.createDescriptorSetLayout(app);
    shadow.createPipeline(app,{image.Count,image.Extent.width,image.Extent.height,RenderPass});
    createDescriptorPool();
}

VkImageView                     & shadowGraphics::getImageView(){return depthAttachment.imageView;}
VkSampler                       & shadowGraphics::getSampler(){return shadowSampler;}
std::vector<VkCommandBuffer>    & shadowGraphics::getCommandBuffer(uint32_t number){return shadowCommandBuffer[number];}

uint32_t                        shadowGraphics::getWidth() const {return image.Extent.width;}
uint32_t                        shadowGraphics::getHeight() const {return image.Extent.height;}
