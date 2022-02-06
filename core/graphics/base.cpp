#include "graphics.h"
#include "core/operations.h"
#include "core/transformational/object.h"
#include "core/transformational/gltfmodel.h"

void graphics::Base::Destroy(VkApplication *app)
{
    vkDestroyPipeline(app->getDevice(), Pipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), PipelineLayout,nullptr);
    vkDestroyDescriptorSetLayout(app->getDevice(), DescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(app->getDevice(), uniformBufferSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(app->getDevice(), uniformBlockSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(app->getDevice(), materialSetLayout, nullptr);
    vkDestroyDescriptorPool(app->getDevice(), DescriptorPool, nullptr);

    for (size_t i = 0; i < uniformBuffers.size(); i++)
    {
        vkDestroyBuffer(app->getDevice(), uniformBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), uniformBuffersMemory[i], nullptr);
    }
}

void graphics::Base::createDescriptorSetLayout(VkApplication *app)
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
        throw std::runtime_error("failed to create descriptor set layout!");
}

void graphics::Base::createPipeline(VkApplication *app, graphicsInfo info)
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
    viewport.width = (float) info.extent.width;
    viewport.height = (float) info.extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = info.extent;

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
    multisampling.rasterizationSamples = info.msaaSamples;
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

        VkPushConstantRange pushConstantRange;
        pushConstantRange.stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstBlockMaterial);

    std::array<VkDescriptorSetLayout,4> SetLayouts = {DescriptorSetLayout,uniformBufferSetLayout,uniformBufferSetLayout,materialSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create pipeline layout!");
    }

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
    pipelineInfo.layout = PipelineLayout;                                   //
    pipelineInfo.renderPass = info.renderPass;                              //проход рендеринга
    pipelineInfo.subpass = 0;                                               //подпроход рендеригка
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &depthStencil;

    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::createBaseDescriptorPool()
{
    /* Наборы дескрипторов нельзя создавать напрямую, они должны выделяться из пула, как буферы команд.
     * Эквивалент для наборов дескрипторов неудивительно называется пулом дескрипторов . Мы напишем
     * новую функцию createDescriptorPool для ее настройки.*/

    {
        size_t index = 0;
        std::vector<VkDescriptorPoolSize> poolSizes(1);
            poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;                           //Сначала нам нужно описать, какие типы дескрипторов будут содержать наши наборы дескрипторов
            poolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);                //и сколько их, используя VkDescriptorPoolSizeструктуры.

        //Мы будем выделять один из этих дескрипторов для каждого кадра. На эту структуру размера пула ссылается главный VkDescriptorPoolCreateInfo:
        VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
            poolInfo.pPoolSizes = poolSizes.data();
            poolInfo.maxSets = static_cast<uint32_t>(imageCount);

        if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &base.DescriptorPool) != VK_SUCCESS)
            throw std::runtime_error("failed to create descriptor pool!");
    }

    for(size_t i=0;i<base.objects.size();i++)
    {
        base.objects.at(i)->setDescriptorSetLayouts({&base.uniformBufferSetLayout,&base.uniformBlockSetLayout,&base.materialSetLayout});
        base.objects.at(i)->createDescriptorPool(imageCount);
    }
    for(size_t i=0;i<bloom.objects.size();i++)
    {
        bloom.objects.at(i)->setDescriptorSetLayouts({&base.uniformBufferSetLayout,&base.uniformBlockSetLayout,&base.materialSetLayout});
        bloom.objects.at(i)->createDescriptorPool(imageCount);
    }
    for(size_t i=0;i<stencil.objects.size();i++)
    {
        stencil.objects.at(i)->setDescriptorSetLayouts({&base.uniformBufferSetLayout,&base.uniformBlockSetLayout,&base.materialSetLayout});
        stencil.objects.at(i)->createDescriptorPool(imageCount);
    }
}

void graphics::createBaseDescriptorSets()
{
    //Теперь мы можем выделить сами наборы дескрипторов
    /* В нашем случае мы создадим один набор дескрипторов для каждого изображения цепочки подкачки, все с одинаковым макетом.
     * К сожалению, нам нужны все копии макета, потому что следующая функция ожидает массив, соответствующий количеству наборов.
     * Добавьте член класса для хранения дескрипторов набора дескрипторов и назначьте их vkAllocateDescriptorSets */

    for(size_t i=0;i<base.objects.size();i++)
        base.objects.at(i)->createDescriptorSet(imageCount);
    for(size_t i=0;i<bloom.objects.size();i++)
        bloom.objects.at(i)->createDescriptorSet(imageCount);
    for(size_t i=0;i<stencil.objects.size();i++)
        stencil.objects.at(i)->createDescriptorSet(imageCount);

    std::vector<VkDescriptorSetLayout> layouts(imageCount, base.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = base.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();

    base.DescriptorSets.resize(imageCount);
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, base.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate descriptor sets!");

    //Наборы дескрипторов уже выделены, но дескрипторы внутри еще нуждаются в настройке.
    //Теперь мы добавим цикл для заполнения каждого дескриптора:
    for (size_t i = 0; i < imageCount; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = base.uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);
        size_t index = 0;
        std::array<VkWriteDescriptorSet, 1> descriptorWrites{};
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = base.DescriptorSets[i];
            descriptorWrites[index].dstBinding = 0;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void graphics::Base::createUniformBuffers(VkApplication *app, uint32_t imageCount)
{
    uniformBuffers.resize(imageCount);
    uniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
    {
        createBuffer(app, sizeof(UniformBufferObject),
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     uniformBuffers[i], uniformBuffersMemory[i]);
    }
}

void graphics::Base::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i, graphics *Graphics)
{
    vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    for(size_t j = 0; j<objects.size() ;j++)
    {
        VkDeviceSize offsets[1] = { 0 };
        vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, & objects[j]->getModel()->vertices.buffer, offsets);
        if (objects[j]->getModel()->indices.buffer != VK_NULL_HANDLE)
            vkCmdBindIndexBuffer(commandBuffers[i], objects[j]->getModel()->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

        for (auto node : objects[j]->getModel()->nodes)
            Graphics->renderNode(node,commandBuffers[i],DescriptorSets[i],objects[j]->getDescriptorSet()[i],PipelineLayout);

//            for (auto node : object3D[j]->getModel()->nodes)
//               if(glm::length(glm::vec3(object3D[j]->getTransformation()*node->matrix*glm::vec4(0.0f,0.0f,0.0f,1.0f))-cameraPosition)<object3D[j]->getVisibilityDistance())
//                    renderNode(node,commandBuffers[i],base.DescriptorSets[i],object3D[j]->getDescriptorSet()[i],*object3D[j]->getPipelineLayout());
    }
}

void graphics::Base::setMaterials(std::vector<PushConstBlockMaterial> &nodeMaterials, graphics *Graphics)
{
    for(size_t j = 0; j<objects.size() ;j++)
        for (auto node : objects[j]->getModel()->nodes)
            Graphics->setMaterialNode(node,nodeMaterials);
}
