#include "shadowGraphics.h"
#include "core/transformational/object.h"
#include "core/operations.h"
#include "core/transformational/gltfmodel.h"

#include <array>

shadowGraphics::shadowGraphics(uint32_t imageCount, VkExtent2D shadowExtent)
{
    image.Count = imageCount;
    image.Extent.width = shadowExtent.width;
    image.Extent.height = shadowExtent.height;
    image.Samples = VK_SAMPLE_COUNT_1_BIT;
}

void shadowGraphics::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, QueueFamilyIndices* queueFamilyIndices)
{
    this->physicalDevice = physicalDevice;
    this->device = device;
    this->queueFamilyIndices = queueFamilyIndices;
    image.Format = findDepthFormat(this->physicalDevice);
}

void shadowGraphics::createMap()
{
    createImage(    physicalDevice,
                    device,
                    image.Extent.width,
                    image.Extent.height,
                    image.Samples,
                    VK_SAMPLE_COUNT_1_BIT,
                    image.Format,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    depthAttachment.image,
                    depthAttachment.imageMemory);
}

void shadowGraphics::createMapView()
{
    depthAttachment.imageView =
            createImageView(    device,
                                depthAttachment.image,
                                image.Format,
                                VK_IMAGE_ASPECT_DEPTH_BIT,
                                image.Samples);
}

void shadowGraphics::destroy()
{
    depthAttachment.deleteAttachment(device);

    shadow.Destroy(device);

    for(uint32_t i=0;i<shadowMapFramebuffer.size();i++)
        vkDestroyFramebuffer(*device, shadowMapFramebuffer.at(i),nullptr);
    vkDestroyRenderPass(*device, RenderPass, nullptr);

    vkFreeCommandBuffers(*device, shadowCommandPool, static_cast<uint32_t>(shadowCommandBuffer.size()), shadowCommandBuffer.data());

    vkDestroyCommandPool(*device, shadowCommandPool, nullptr);

    vkDestroySampler(*device, shadowSampler, nullptr);
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
        samplerInfo.minLod = static_cast<float>(mipLevel*image.Samples);
        samplerInfo.maxLod = static_cast<float>(image.Samples);
        samplerInfo.mipLodBias = 0.0f;
    if (vkCreateSampler(*device, &samplerInfo, nullptr, &shadowSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadowGraphics sampler!");
}

void shadowGraphics::createRenderPass()
{
    VkAttachmentDescription attachments{};
        attachments.format =  findDepthFormat(physicalDevice);
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

    vkCreateRenderPass(*device, &renderPassInfo, NULL, &RenderPass);
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
        vkCreateFramebuffer(*device, &framebufferInfo, NULL, &shadowMapFramebuffer.at(i));
    }
}

void shadowGraphics::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices->graphicsFamily.value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(*device, &poolInfo, nullptr, &shadowCommandPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadowGraphics command pool!");
}

void shadowGraphics::Shadow::Destroy(VkDevice* device)
{
    vkDestroyPipeline(*device, Pipeline, nullptr);
    vkDestroyPipelineLayout(*device, PipelineLayout,nullptr);
    vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(*device, uniformBlockSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(*device, uniformBufferSetLayout, nullptr);
    vkDestroyDescriptorPool(*device, DescriptorPool, nullptr);
}

void shadowGraphics::Shadow::createPipeline(VkDevice* device, shadowInfo info)
{
    const std::string ExternalPath = "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\";
    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\shadow\\shad.spv");
    VkShaderModule vertShaderModule = createShaderModule(device,vertShaderCode);

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
    viewport.width = (float) info.extent.width;
    viewport.height = (float) info.extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent.width = info.extent.width;
    scissor.extent.height = info.extent.height;
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

    std::array<VkDescriptorSetLayout,3> SetLayouts = {DescriptorSetLayout,uniformBufferSetLayout,uniformBlockSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();

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

    if (vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadowGraphics pipeline layout!");

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
    if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadowGraphics graphics pipeline!");

    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
}

void shadowGraphics::Shadow::createDescriptorSetLayout(VkDevice* device)
{
    VkDescriptorSetLayoutBinding lightUboLayoutBinding={};
        lightUboLayoutBinding.binding = 0;
        lightUboLayoutBinding.descriptorCount = 1;
        lightUboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        lightUboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        lightUboLayoutBinding.pImmutableSamplers = nullptr;
    std::array<VkDescriptorSetLayoutBinding, 1> bindings = {lightUboLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(*device, &layoutInfo, nullptr, &DescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadowGraphics descriptor set layout!");

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
    if (vkCreateDescriptorSetLayout(*device, &uniformBufferLayoutInfo, nullptr, &uniformBufferSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadowGraphics uniformb buffer descriptor set layout!");

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
    if (vkCreateDescriptorSetLayout(*device, &uniformBlockLayoutInfo, nullptr, &uniformBlockSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadowGraphics uniformb block descriptor set layout!");
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

    if (vkCreateDescriptorPool(*device, &poolInfo, nullptr, &shadow.DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadowGraphics descriptor pool!");
}

void shadowGraphics::createDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> layouts(image.Count, shadow.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = shadow.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();

    shadow.DescriptorSets.resize(image.Count);
    if (vkAllocateDescriptorSets(*device, &allocInfo, shadow.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate shadowGraphics descriptor sets!");
}

void shadowGraphics::updateDescriptorSets(uint32_t lightUniformBuffersCount, VkBuffer* plightUniformBuffers, unsigned long long sizeOfLightUniformBuffers)
{
    for (size_t i=0; i<lightUniformBuffersCount; i++)
    {
        VkDescriptorBufferInfo lightBufferInfo;
            lightBufferInfo.buffer = plightUniformBuffers[i];
            lightBufferInfo.offset = 0;
            lightBufferInfo.range = sizeOfLightUniformBuffers;
        VkWriteDescriptorSet descriptorWrites{};
            descriptorWrites.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.dstSet = shadow.DescriptorSets[i];
            descriptorWrites.dstBinding = 0;
            descriptorWrites.dstArrayElement = 0;
            descriptorWrites.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.descriptorCount = 1;
            descriptorWrites.pBufferInfo = &lightBufferInfo;
        vkUpdateDescriptorSets(*device, 1, &descriptorWrites, 0, nullptr);
    }
}

void shadowGraphics::createCommandBuffers()
{
    shadowCommandBuffer.resize(image.Count);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = shadowCommandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = image.Count;
    if (vkAllocateCommandBuffers(*device, &allocInfo, shadowCommandBuffer.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate shadowGraphics command buffers!");
}

void shadowGraphics::updateCommandBuffer(uint32_t frameNumber, ShadowPassObjects objects)
{
    VkClearValue clearValues{};
        clearValues.depthStencil.depth = 1.0f;
        clearValues.depthStencil.stencil = 0;

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;                                            //поле для передачи информации о том, как будет использоваться этот командный буфер (смотри страницу 102)
        beginInfo.pInheritanceInfo = nullptr;                           //используется при начале вторичного буфера, для того чтобы определить, какие состояния наследуются от первичного командного буфера, который его вызовет
    if (vkBeginCommandBuffer(shadowCommandBuffer[frameNumber], &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording shadowGraphics command buffer!");

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.pNext = NULL;
        renderPassInfo.renderPass = RenderPass;
        renderPassInfo.framebuffer = shadowMapFramebuffer[frameNumber];
        renderPassInfo.renderArea.offset.x = 0;
        renderPassInfo.renderArea.offset.y = 0;
        renderPassInfo.renderArea.extent.width = image.Extent.width;
        renderPassInfo.renderArea.extent.height = image.Extent.height;
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearValues;

    vkCmdBeginRenderPass(shadowCommandBuffer[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport;
            viewport.width = image.Extent.width;
            viewport.height = image.Extent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            viewport.x = 0;
            viewport.y = 0;
        vkCmdSetViewport(shadowCommandBuffer[frameNumber], 0, 1, &viewport);

        VkRect2D scissor;
            scissor.extent.width = image.Extent.width;
            scissor.extent.height = image.Extent.height;
            scissor.offset.x = 0;
            scissor.offset.y = 0;
        vkCmdSetScissor(shadowCommandBuffer[frameNumber], 0, 1, &scissor);

        vkCmdBindPipeline(shadowCommandBuffer[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.Pipeline);
        for(auto object: *objects.base)
        {
            if(object->getEnable()){
                VkDeviceSize offsets[1] = { 0 };
                vkCmdBindVertexBuffers(shadowCommandBuffer[frameNumber], 0, 1, & object->getModel(frameNumber)->vertices.buffer, offsets);
                if (object->getModel(frameNumber)->indices.buffer != VK_NULL_HANDLE)
                    vkCmdBindIndexBuffer(shadowCommandBuffer[frameNumber],  object->getModel(frameNumber)->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

                for (auto node : object->getModel(frameNumber)->nodes){
                    std::vector<VkDescriptorSet> descriptorSets = {shadow.DescriptorSets[frameNumber],object->getDescriptorSet()[frameNumber]};
                    renderNode(shadowCommandBuffer[frameNumber],node,static_cast<uint32_t>(descriptorSets.size()),descriptorSets.data());
                }
            }
        }
        for(auto object: *objects.oneColor)
        {
            if(object->getEnable()){
                VkDeviceSize offsets[1] = { 0 };
                vkCmdBindVertexBuffers(shadowCommandBuffer[frameNumber], 0, 1, & object->getModel(frameNumber)->vertices.buffer, offsets);

                if (object->getModel(frameNumber)->indices.buffer != VK_NULL_HANDLE)
                    vkCmdBindIndexBuffer(shadowCommandBuffer[frameNumber],  object->getModel(frameNumber)->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

                for (auto node : object->getModel(frameNumber)->nodes){
                    std::vector<VkDescriptorSet> descriptorSets = {shadow.DescriptorSets[frameNumber],object->getDescriptorSet()[frameNumber]};
                    renderNode(shadowCommandBuffer[frameNumber],node,static_cast<uint32_t>(descriptorSets.size()),descriptorSets.data());
                }
            }
        }
        for(auto object: *objects.stencil)
        {
            if(object->getEnable()){
                VkDeviceSize offsets[1] = { 0 };
                vkCmdBindVertexBuffers(shadowCommandBuffer[frameNumber], 0, 1, & object->getModel(frameNumber)->vertices.buffer, offsets);

                if (object->getModel(frameNumber)->indices.buffer != VK_NULL_HANDLE)
                    vkCmdBindIndexBuffer(shadowCommandBuffer[frameNumber],  object->getModel(frameNumber)->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

                for (auto node : object->getModel(frameNumber)->nodes){
                    std::vector<VkDescriptorSet> descriptorSets = {shadow.DescriptorSets[frameNumber],object->getDescriptorSet()[frameNumber]};
                    renderNode(shadowCommandBuffer[frameNumber],node,static_cast<uint32_t>(descriptorSets.size()),descriptorSets.data());
                }
            }
        }

    vkCmdEndRenderPass(shadowCommandBuffer[frameNumber]);

    if (vkEndCommandBuffer(shadowCommandBuffer[frameNumber]) != VK_SUCCESS)
        throw std::runtime_error("failed to record shadowGraphics command buffer!");
}

void shadowGraphics::renderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets)
{
    if (node->mesh)
    {
        std::vector<VkDescriptorSet> nodeDescriptorSets(descriptorSetsCount+1);
        for(uint32_t i=0;i<descriptorSetsCount;i++)
            nodeDescriptorSets[i] = descriptorSets[i];
        nodeDescriptorSets[descriptorSetsCount] = node->mesh->uniformBuffer.descriptorSet;

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.PipelineLayout, 0, descriptorSetsCount+1, nodeDescriptorSets.data(), 0, NULL);

        for (Primitive* primitive : node->mesh->primitives)
            if (primitive->hasIndices)  vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
            else                        vkCmdDraw(commandBuffer, primitive->vertexCount, 1, 0, 0);
    }
    for (auto child : node->children)
        renderNode(commandBuffer, child, descriptorSetsCount, descriptorSets);
}

void shadowGraphics::createShadow()
{
    createCommandPool();
    createMap();
    createMapView();
    createSampler();
    createRenderPass();
    createFramebuffer();
    shadow.createDescriptorSetLayout(device);
    shadow.createPipeline(device,{image.Count,image.Extent,VK_SAMPLE_COUNT_1_BIT,RenderPass});
    createDescriptorPool();
    createDescriptorSets();
}

VkImageView                     & shadowGraphics::getImageView(){return depthAttachment.imageView;}
VkSampler                       & shadowGraphics::getSampler(){return shadowSampler;}
std::vector<VkCommandBuffer>    & shadowGraphics::getCommandBuffer(){return shadowCommandBuffer;}

uint32_t                        shadowGraphics::getWidth() const {return image.Extent.width;}
uint32_t                        shadowGraphics::getHeight() const {return image.Extent.height;}
