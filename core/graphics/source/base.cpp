#include "core/graphics/graphics.h"
#include "core/operations.h"
#include "core/transformational/object.h"
#include "core/transformational/gltfmodel.h"
#include <array>

void deferredGraphics::Base::Destroy(VkDevice* device)
{
    vkDestroyPipeline(*device, Pipeline, nullptr);
    vkDestroyPipelineLayout(*device, PipelineLayout,nullptr);
    vkDestroyDescriptorSetLayout(*device, SceneDescriptorSetLayout,  nullptr);
    vkDestroyDescriptorSetLayout(*device, ObjectDescriptorSetLayout,  nullptr);
    vkDestroyDescriptorSetLayout(*device, PrimitiveDescriptorSetLayout,  nullptr);
    vkDestroyDescriptorSetLayout(*device, MaterialDescriptorSetLayout,  nullptr);
    vkDestroyDescriptorPool(*device, DescriptorPool, nullptr);

    for (size_t i = 0; i < sceneUniformBuffers.size(); i++)
    {
        vkDestroyBuffer(*device, sceneUniformBuffers[i], nullptr);
        vkFreeMemory(*device, sceneUniformBuffersMemory[i], nullptr);
    }
}

void deferredGraphics::Base::createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount)
{
    sceneUniformBuffers.resize(imageCount);
    sceneUniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
    {
        createBuffer(   physicalDevice,
                        device,
                        sizeof(UniformBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        sceneUniformBuffers[i],
                        sceneUniformBuffersMemory[i]);
    }
}

void deferredGraphics::Base::createDescriptorSetLayout(VkDevice* device)
{
    uint32_t index = 0;

    std::array<VkDescriptorSetLayoutBinding, 3> Binding{};
        Binding[index].binding = index;
        Binding[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        Binding[index].descriptorCount = 1;
        Binding[index].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        Binding[index].pImmutableSamplers = nullptr;
    index++;
        Binding[index].binding = index;
        Binding[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        Binding[index].descriptorCount = 1;
        Binding[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        Binding[index].pImmutableSamplers = nullptr;
    index++;
        Binding.at(index).binding = index;
        Binding.at(index).descriptorCount = 1;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(Binding.size());
        layoutInfo.pBindings = Binding.data();
    if (vkCreateDescriptorSetLayout(*device, &layoutInfo, nullptr, &SceneDescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create base uniform buffer descriptor set layout!");

    index = 0;
    std::array<VkDescriptorSetLayoutBinding, 1> uniformBufferLayoutBinding{};
        uniformBufferLayoutBinding[index].binding = 0;
        uniformBufferLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniformBufferLayoutBinding[index].descriptorCount = 1;
        uniformBufferLayoutBinding[index].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uniformBufferLayoutBinding[index].pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo uniformBufferLayoutInfo{};
        uniformBufferLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        uniformBufferLayoutInfo.bindingCount = static_cast<uint32_t>(uniformBufferLayoutBinding.size());
        uniformBufferLayoutInfo.pBindings = uniformBufferLayoutBinding.data();
    if (vkCreateDescriptorSetLayout(*device, &uniformBufferLayoutInfo, nullptr, &ObjectDescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create base object uniform buffer descriptor set layout!");

    index = 0;
    std::array<VkDescriptorSetLayoutBinding, 1> uniformBlockLayoutBinding{};
        uniformBlockLayoutBinding[index].binding = 0;
        uniformBlockLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniformBlockLayoutBinding[index].descriptorCount = 1;
        uniformBlockLayoutBinding[index].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uniformBlockLayoutBinding[index].pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo uniformBlockLayoutInfo{};
        uniformBlockLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        uniformBlockLayoutInfo.bindingCount = static_cast<uint32_t>(uniformBlockLayoutBinding.size());
        uniformBlockLayoutInfo.pBindings = uniformBlockLayoutBinding.data();
    if (vkCreateDescriptorSetLayout(*device, &uniformBlockLayoutInfo, nullptr, &PrimitiveDescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create base uniform block descriptor set layout!");

    index = 0;
    std::array<VkDescriptorSetLayoutBinding, 5> materialLayoutBinding{};
    //baseColorTexture;
        materialLayoutBinding[index].binding = 0;
        materialLayoutBinding[index].descriptorCount = 1;
        materialLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        materialLayoutBinding[index].pImmutableSamplers = nullptr;
        materialLayoutBinding[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    index++;
    //metallicRoughnessTexture;
        materialLayoutBinding[index].binding = 1;
        materialLayoutBinding[index].descriptorCount = 1;
        materialLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        materialLayoutBinding[index].pImmutableSamplers = nullptr;
        materialLayoutBinding[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    index++;
    //normalTexture;
        materialLayoutBinding[index].binding = 2;
        materialLayoutBinding[index].descriptorCount = 1;
        materialLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        materialLayoutBinding[index].pImmutableSamplers = nullptr;
        materialLayoutBinding[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    index++;
    //occlusionTexture;
        materialLayoutBinding[index].binding = 3;
        materialLayoutBinding[index].descriptorCount = 1;
        materialLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        materialLayoutBinding[index].pImmutableSamplers = nullptr;
        materialLayoutBinding[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    index++;
    //emissiveTexture;
        materialLayoutBinding[index].binding = 4;
        materialLayoutBinding[index].descriptorCount = 1;
        materialLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        materialLayoutBinding[index].pImmutableSamplers = nullptr;
        materialLayoutBinding[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo materialLayoutInfo{};
        materialLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        materialLayoutInfo.bindingCount = static_cast<uint32_t>(materialLayoutBinding.size());
        materialLayoutInfo.pBindings = materialLayoutBinding.data();
    if (vkCreateDescriptorSetLayout(*device, &materialLayoutInfo, nullptr, &MaterialDescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create base material descriptor set layout!");
}

void deferredGraphics::Base::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    uint32_t index = 0;

    const std::string ExternalPath = "C:\\Users\\kiril\\OneDrive\\qt\\kisskaVulkan\\";
    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\base\\basevert.spv");   //считываем шейдеры
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\base\\basefrag.spv");
    VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);                      //создаём шейдерные модули
    VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);
    std::array<VkPipelineShaderStageCreateInfo,2> shaderStages{};                           //задаём стадии шейдеров в конвейере
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;    //вершинный
        shaderStages[index].stage = VK_SHADER_STAGE_VERTEX_BIT;                             //ниформацию о всех битах смотри на странице 222
        shaderStages[index].module = vertShaderModule;                                      //сюда передаём шейдерный модуль
        shaderStages[index].pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа
    index++;
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;    //фрагментный
        shaderStages[index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;                           //ниформацию о всех битах смотри на странице 222
        shaderStages[index].module = fragShaderModule;                                      //сюда передаём шейдерный модуль
        shaderStages[index].pName = "main";                                                 //указатель на строку UTF-8 с завершающим нулем, определяющую имя точки входа шейдера для этого этапа

    /* Для рендеринга настоящей геометрии вам необходимо передавать данные в конвайер Vulkan.
     * Вы можете использовать индексы вершин и экземпляров, доступные в SPIR-V, для автоматической
     * генерации геометрии или же явно извлекать геометрические данные из буфера. Вместо этого вы можете
     * описать размещение геометрических данных в памяти, и Vulkan может сам извлекать эти данные для вас, передавая их прямо в шейдер*/

    auto bindingDescription = gltfModel::Vertex::getBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    /* фаза входной сборки графического конвейера берёт данные в вершинах и группирует их в примитивы,
     * готовые для обработки следубщими стадиями конвейера.*/

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

    /* здесь может быть добавлена тесселяция*/

    /* Преобразование области вывода - это последнее преобразование координат в конвейере Vulkan до растретизации.
     * Оно преобразует координаты вершины из нормализованных координат устройства в оконные координаты. Одновременно
     * может использоваться несколько областей вывода.*/

    index = 0;
    std::array<VkViewport,1> viewport{};
        viewport[index].x = 0.0f;
        viewport[index].y = 0.0f;
        viewport[index].width = (float) pInfo->Extent.width;
        viewport[index].height = (float) pInfo->Extent.height;
        viewport[index].minDepth = 0.0f;
        viewport[index].maxDepth = 1.0f;
    std::array<VkRect2D,1> scissor{};
        scissor[index].offset = {0, 0};
        scissor[index].extent = pInfo->Extent;
    VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = static_cast<uint32_t>(viewport.size());;              //число областей вывода
        viewportState.pViewports = viewport.data();                                         //размер каждой области вывода
        viewportState.scissorCount = static_cast<uint32_t>(scissor.size());;                //число прямоугольников
        viewportState.pScissors = scissor.data();                                           //эксцент

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
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

    /* Мультсемплинг - это процесс создания нескольких образцов (sample) для каждого пиксела в изображении.
     * Они используются для борьбы с алиансингом и может заметно улучшить общее качество изображения при эффективном использовании*/

    VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = pInfo->Samples;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

    /* Последней стадией в графическом конвейере является стадия смешивания цветов. Эта стадия отвечает за запись фрагментов
     * в цветовые подключения. Во многих случаях это простая операция, которая просто записывает содержимое выходного значения
     * фрагментного шейдера поверх старого значения. Однакоподдеживаются смешивание этих значнеий со значениями,
     * уже находящимися во фрейм буфере, и выполнение простых логических операций между выходными значениями фрагментного
     * шейдера и текущим содержанием фреймбуфера.*/

    std::array<VkPipelineColorBlendAttachmentState,4> colorBlendAttachment;
    for(index=0;index<colorBlendAttachment.size();index++)
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

    index=0;
    std::array<VkPushConstantRange,1> pushConstantRange{};
        pushConstantRange[index].stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange[index].offset = 0;
        pushConstantRange[index].size = sizeof(MaterialBlock);
    std::array<VkDescriptorSetLayout,4> setLayouts = {SceneDescriptorSetLayout,ObjectDescriptorSetLayout,PrimitiveDescriptorSetLayout,MaterialDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
        pipelineLayoutInfo.pSetLayouts = setLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    if (vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create base pipeline layout!");

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};

    index=0;
    std::array<VkGraphicsPipelineCreateInfo,1> pipelineInfo{};
        pipelineInfo[index].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo[index].stageCount = static_cast<uint32_t>(shaderStages.size());   //число структур в массиве структур
        pipelineInfo[index].pStages = shaderStages.data();                             //указывает на массив структур VkPipelineShaderStageCreateInfo, каждая из которых описыват одну стадию
        pipelineInfo[index].pVertexInputState = &vertexInputInfo;                      //вершинный ввод
        pipelineInfo[index].pInputAssemblyState = &inputAssembly;                      //фаза входной сборки
        pipelineInfo[index].pViewportState = &viewportState;                           //Преобразование области вывода
        pipelineInfo[index].pRasterizationState = &rasterizer;                         //растеризация
        pipelineInfo[index].pMultisampleState = &multisampling;                        //мультсемплинг
        pipelineInfo[index].pColorBlendState = &colorBlending;                         //смешивание цветов
        pipelineInfo[index].layout = PipelineLayout;                                   //
        pipelineInfo[index].renderPass = *pRenderPass;                                 //проход рендеринга
        pipelineInfo[index].subpass = 0;                                               //подпроход рендеригка
        pipelineInfo[index].pDepthStencilState = &depthStencil;
        pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create base graphics pipeline!");

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(*device, fragShaderModule, nullptr);
    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
}

void deferredGraphics::createBaseDescriptorPool()
{
    /* Наборы дескрипторов нельзя создавать напрямую, они должны выделяться из пула, как буферы команд.
     * Эквивалент для наборов дескрипторов неудивительно называется пулом дескрипторов . Мы напишем
     * новую функцию createDescriptorPool для ее настройки.*/

    uint32_t index = 0;

    std::array<VkDescriptorPoolSize,4> poolSizes;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(image.Count);
    index++;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(image.Count);
    index++;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(image.Count);    
    index++;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(image.Count);
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    if (vkCreateDescriptorPool(*device, &poolInfo, nullptr, &base.DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create base descriptor pool!");
}

void deferredGraphics::createBaseDescriptorSets()
{
    //Теперь мы можем выделить сами наборы дескрипторов
    /* В нашем случае мы создадим один набор дескрипторов для каждого изображения цепочки подкачки, все с одинаковым макетом.
     * К сожалению, нам нужны все копии макета, потому что следующая функция ожидает массив, соответствующий количеству наборов.
     * Добавьте член класса для хранения дескрипторов набора дескрипторов и назначьте их vkAllocateDescriptorSets */

    base.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, base.SceneDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = base.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(*device, &allocInfo, base.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate base descriptor sets!");
}

void deferredGraphics::updateBaseDescriptorSets()
{
    for (size_t i = 0; i < image.Count; i++)
    {
        uint32_t index = 0;

        std::array<VkDescriptorBufferInfo,1> bufferInfo{};
            bufferInfo[0].buffer = base.sceneUniformBuffers[i];
            bufferInfo[0].offset = 0;
            bufferInfo[0].range = sizeof(UniformBufferObject);
        std::array<VkDescriptorImageInfo,1> skyboxImageInfo{};
            skyboxImageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            skyboxImageInfo[0].imageView = skybox.texture ? skybox.texture->getTextureImageView() : emptyTexture->getTextureImageView();
            skyboxImageInfo[0].sampler   = skybox.texture ? skybox.texture->getTextureSampler() : emptyTexture->getTextureSampler();
        VkDescriptorBufferInfo StorageBufferInfo{};
            StorageBufferInfo.buffer = storageBuffers[i];
            StorageBufferInfo.offset = 0;
            StorageBufferInfo.range = sizeof(StorageBufferObject);

        std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = base.DescriptorSets[i];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[index].descriptorCount = static_cast<uint32_t>(bufferInfo.size());
            descriptorWrites[index].pBufferInfo = bufferInfo.data();
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = base.DescriptorSets[i];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = static_cast<uint32_t>(skyboxImageInfo.size());
            descriptorWrites[index].pImageInfo = skyboxImageInfo.data();
        index++;
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = base.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pBufferInfo = &StorageBufferInfo;
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void deferredGraphics::Base::createObjectDescriptorPool(VkDevice* device, object* object, uint32_t imageCount)
{
    size_t index = 0;
    std::vector<VkDescriptorPoolSize> DescriptorPoolSizes(1);
        DescriptorPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        DescriptorPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);

    //Мы будем выделять один из этих дескрипторов для каждого кадра. На эту структуру размера пула ссылается главный VkDescriptorPoolCreateInfo:
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(DescriptorPoolSizes.size());
    poolInfo.pPoolSizes = DescriptorPoolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(imageCount);

    if (vkCreateDescriptorPool(*device, &poolInfo, nullptr, &object->getDescriptorPool()) != VK_SUCCESS)
        throw std::runtime_error("failed to create object descriptor pool!");
}

void deferredGraphics::Base::createObjectDescriptorSet(VkDevice* device, object* object, uint32_t imageCount)
{
    std::vector<VkDescriptorSetLayout> layouts(imageCount, ObjectDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = object->getDescriptorPool();
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    object->getDescriptorSet().resize(imageCount);
    if (vkAllocateDescriptorSets(*device, &allocInfo, object->getDescriptorSet().data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate object descriptor sets!");

    for (size_t i = 0; i < imageCount; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = object->getUniformBuffers()[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBuffer);
        VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.dstSet = object->getDescriptorSet()[i];
            writeDescriptorSet.dstBinding = 0;
            writeDescriptorSet.pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(*device, 1, &writeDescriptorSet, 0, nullptr);
    }
}

void deferredGraphics::Base::createModelDescriptorPool(VkDevice* device, gltfModel* pModel)
{
    uint32_t imageSamplerCount = 0;
    uint32_t materialCount = 0;
    uint32_t meshCount = 0;
    for (auto &material : pModel->materials)
    {
        static_cast<void>(material);
        imageSamplerCount += 5;
        materialCount++;
    }

    for (auto node : pModel->linearNodes){
        if(node->mesh){
            meshCount++;
        }
    }

    size_t index = 0;
    std::vector<VkDescriptorPoolSize> DescriptorPoolSizes(2);
        DescriptorPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        DescriptorPoolSizes.at(index).descriptorCount = meshCount;
    index++;
        DescriptorPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        DescriptorPoolSizes.at(index).descriptorCount = imageSamplerCount;
    index++;

    //Мы будем выделять один из этих дескрипторов для каждого кадра. На эту структуру размера пула ссылается главный VkDescriptorPoolCreateInfo:
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(DescriptorPoolSizes.size());
        poolInfo.pPoolSizes = DescriptorPoolSizes.data();
        poolInfo.maxSets = meshCount+imageSamplerCount;
    if (vkCreateDescriptorPool(*device, &poolInfo, nullptr, &pModel->DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create object descriptor pool!");
}
void deferredGraphics::Base::createModelDescriptorSet(VkDevice* device, gltfModel* pModel, texture* emptyTexture)
{
    for (auto node : pModel->linearNodes){
        if(node->mesh){
            createModelNodeDescriptorSet(device, pModel, node);
        }
    }

    for (auto &material : pModel->materials){
        createModelMaterialDescriptorSet(device, pModel, &material, emptyTexture);
    }
}
void deferredGraphics::Base::createModelNodeDescriptorSet(VkDevice* device, gltfModel* pModel, Node* node)
{
    if (node->mesh)
    {
        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = pModel->DescriptorPool;
            descriptorSetAllocInfo.pSetLayouts = &PrimitiveDescriptorSetLayout;
            descriptorSetAllocInfo.descriptorSetCount = 1;
        if (vkAllocateDescriptorSets(*device, &descriptorSetAllocInfo, &node->mesh->uniformBuffer.descriptorSet) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate object descriptor sets!");

        VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.dstSet = node->mesh->uniformBuffer.descriptorSet;
            writeDescriptorSet.dstBinding = 0;
            writeDescriptorSet.pBufferInfo = &node->mesh->uniformBuffer.descriptor;
        vkUpdateDescriptorSets(*device, 1, &writeDescriptorSet, 0, nullptr);
    }
    for (auto& child : node->children)
        createModelNodeDescriptorSet(device,pModel,child);
}

void deferredGraphics::Base::createModelMaterialDescriptorSet(VkDevice* device, gltfModel* pModel, Material* material, texture* emptyTexture)
{
    std::vector<VkDescriptorSetLayout> layouts(1, MaterialDescriptorSetLayout);
    VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = pModel->DescriptorPool;
        descriptorSetAllocInfo.pSetLayouts = layouts.data();
        descriptorSetAllocInfo.descriptorSetCount = 1;
    if (vkAllocateDescriptorSets(*device, &descriptorSetAllocInfo, &material->descriptorSet) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate object descriptor sets!");

    VkDescriptorImageInfo baseColorTextureInfo;
    baseColorTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (material->pbrWorkflows.metallicRoughness)
    {
        baseColorTextureInfo.imageView   = material->baseColorTexture ? material->baseColorTexture->getTextureImageView() : emptyTexture->getTextureImageView();
        baseColorTextureInfo.sampler     = material->baseColorTexture ? material->baseColorTexture->getTextureSampler()   : emptyTexture->getTextureSampler();
    }
    if(material->pbrWorkflows.specularGlossiness)
    {
        baseColorTextureInfo.imageView   = material->extension.diffuseTexture ? material->extension.diffuseTexture->getTextureImageView() : emptyTexture->getTextureImageView();
        baseColorTextureInfo.sampler     = material->extension.diffuseTexture ? material->extension.diffuseTexture->getTextureSampler() : emptyTexture->getTextureSampler();
    }

    VkDescriptorImageInfo metallicRoughnessTextureInfo;
    metallicRoughnessTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (material->pbrWorkflows.metallicRoughness)
    {
        metallicRoughnessTextureInfo.imageView   = material->metallicRoughnessTexture ? material->metallicRoughnessTexture->getTextureImageView() : emptyTexture->getTextureImageView();
        metallicRoughnessTextureInfo.sampler     = material->metallicRoughnessTexture ? material->metallicRoughnessTexture->getTextureSampler() : emptyTexture->getTextureSampler();
    }
    if (material->pbrWorkflows.specularGlossiness)
    {
        metallicRoughnessTextureInfo.imageView   = material->extension.specularGlossinessTexture ? material->extension.specularGlossinessTexture->getTextureImageView() : emptyTexture->getTextureImageView();
        metallicRoughnessTextureInfo.sampler     = material->extension.specularGlossinessTexture ? material->extension.specularGlossinessTexture->getTextureSampler() : emptyTexture->getTextureSampler();
    }

    VkDescriptorImageInfo normalTextureInfo;
    normalTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    normalTextureInfo.imageView   = material->normalTexture ? material->normalTexture->getTextureImageView() : emptyTexture->getTextureImageView();
    normalTextureInfo.sampler     = material->normalTexture ? material->normalTexture->getTextureSampler() : emptyTexture->getTextureSampler();

    VkDescriptorImageInfo occlusionTextureInfo;
    occlusionTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    occlusionTextureInfo.imageView   = material->occlusionTexture ? material->occlusionTexture->getTextureImageView() : emptyTexture->getTextureImageView();
    occlusionTextureInfo.sampler     = material->occlusionTexture ? material->occlusionTexture->getTextureSampler() : emptyTexture->getTextureSampler();

    VkDescriptorImageInfo emissiveTextureInfo;
    emissiveTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    emissiveTextureInfo.imageView   = material->emissiveTexture ? material->emissiveTexture->getTextureImageView() : emptyTexture->getTextureImageView();
    emissiveTextureInfo.sampler     = material->emissiveTexture ? material->emissiveTexture->getTextureSampler() : emptyTexture->getTextureSampler();

    std::array<VkDescriptorImageInfo, 5> descriptorImageInfos = {baseColorTextureInfo,metallicRoughnessTextureInfo,normalTextureInfo,occlusionTextureInfo,emissiveTextureInfo};
    std::array<VkWriteDescriptorSet, 5> descriptorWrites{};

    for(size_t i=0;i<5;i++)
    {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = material->descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pImageInfo = &descriptorImageInfos[i];
    }
    vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void deferredGraphics::Base::render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount)
{
    vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    for(auto object: objects)
    {
        if(object->getEnable()){
            VkDeviceSize offsets[1] = { 0 };
            vkCmdBindVertexBuffers(commandBuffers, 0, 1, & object->getModel(frameNumber)->vertices.buffer, offsets);
            if (object->getModel(frameNumber)->indices.buffer != VK_NULL_HANDLE)
                vkCmdBindIndexBuffer(commandBuffers, object->getModel(frameNumber)->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

            object->resetPrimitiveCount();
            object->setFirstPrimitive(primitiveCount);
            for (auto node : object->getModel(frameNumber)->nodes){
                std::vector<VkDescriptorSet> descriptorSets = {DescriptorSets[frameNumber],object->getDescriptorSet()[frameNumber]};
                renderNode(commandBuffers,node,static_cast<uint32_t>(descriptorSets.size()),descriptorSets.data(), primitiveCount);
            }
            object->setPrimitiveCount(primitiveCount-object->getFirstPrimitive());
        }
    }
}

void deferredGraphics::Base::renderNode(VkCommandBuffer commandBuffer, Node *node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount)
{
    if (node->mesh)
    {
        for (Primitive* primitive : node->mesh->primitives)
        {
            std::vector<VkDescriptorSet> nodeDescriptorSets(descriptorSetsCount+2);
            for(uint32_t i=0;i<descriptorSetsCount;i++)
                nodeDescriptorSets[i] = descriptorSets[i];
            nodeDescriptorSets[descriptorSetsCount+0] = node->mesh->uniformBuffer.descriptorSet;
            nodeDescriptorSets[descriptorSetsCount+1] = primitive->material.descriptorSet;

            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, descriptorSetsCount+2, nodeDescriptorSets.data(), 0, NULL);

            MaterialBlock pushConstBlockMaterial{};

            pushConstBlockMaterial.emissiveFactor = primitive->material.emissiveFactor;
            // To save push constant space, availabilty and texture coordiante set are combined
            // -1 = texture not used for this material, >= 0 texture used and index of texture coordinate set
            pushConstBlockMaterial.colorTextureSet = primitive->material.baseColorTexture != nullptr ? primitive->material.texCoordSets.baseColor : -1;
            pushConstBlockMaterial.normalTextureSet = primitive->material.normalTexture != nullptr ? primitive->material.texCoordSets.normal : -1;
            pushConstBlockMaterial.occlusionTextureSet = primitive->material.occlusionTexture != nullptr ? primitive->material.texCoordSets.occlusion : -1;
            pushConstBlockMaterial.emissiveTextureSet = primitive->material.emissiveTexture != nullptr ? primitive->material.texCoordSets.emissive : -1;
            pushConstBlockMaterial.alphaMask = static_cast<float>(primitive->material.alphaMode == Material::ALPHAMODE_MASK);
            pushConstBlockMaterial.alphaMaskCutoff = primitive->material.alphaCutoff;

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

            pushConstBlockMaterial.primitive = primitiveCount;

            vkCmdPushConstants(commandBuffer, PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(MaterialBlock), &pushConstBlockMaterial);

            if (primitive->hasIndices)
                vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
            else
                vkCmdDraw(commandBuffer, primitive->vertexCount, 1, 0, 0);

            primitiveCount++;
        }
    }
    for (auto child : node->children)
        renderNode(commandBuffer,child,descriptorSetsCount,descriptorSets,primitiveCount);
}
