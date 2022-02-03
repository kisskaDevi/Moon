#include "graphics.h"
#include "core/operations.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/light.h"

void graphics::Second::Destroy(VkApplication *app)
{
    vkDestroyDescriptorSetLayout(app->getDevice(), DescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(app->getDevice(), DescriptorPool, nullptr);
    vkDestroyPipeline(app->getDevice(), Pipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), PipelineLayout, nullptr);

    for (size_t i = 0; i < uniformBuffers.size(); i++)
    {
        vkDestroyBuffer(app->getDevice(), lightUniformBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), lightUniformBuffersMemory[i], nullptr);

        vkDestroyBuffer(app->getDevice(), uniformBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), uniformBuffersMemory[i], nullptr);

        vkDestroyBuffer(app->getDevice(), emptyUniformBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), emptyUniformBuffersMemory[i], nullptr);
    }
}

void graphics::Second::createDescriptorSetLayout(VkApplication *app)
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
        Binding.at(index).descriptorCount = 1;
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

    if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &DescriptorSetLayout) != VK_SUCCESS)
    {throw std::runtime_error("failed to create descriptor set layout!");}
}

void graphics::Second::createPipeline(VkApplication *app, graphicsInfo info)
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
    vertexInputInfo.vertexBindingDescriptionCount = 0;                                          //количество привязанных дескрипторов вершин
    vertexInputInfo.vertexAttributeDescriptionCount = 0;                                        //количество дескрипторов атрибутов вершин
    vertexInputInfo.pVertexBindingDescriptions = nullptr;                                       //указатель на массив соответствующийх структуру
    vertexInputInfo.pVertexAttributeDescriptions = nullptr;                                     //указатель на массив соответствующийх структуру

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
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;                                     //параметр направления обхода (против часовой стрелки)
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

    std::array<VkDescriptorSetLayout,1> SetLayouts = {DescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout!");

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
    pipelineInfo.layout = PipelineLayout;
    pipelineInfo.renderPass = info.renderPass;                              //проход рендеринга
    pipelineInfo.subpass = 1;                                               //подпроход рендеригка
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &depthStencil;

    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics pipeline!");

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::createSecondDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    for(size_t i=0;i<6;i++)
        poolSizes.push_back({VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, static_cast<uint32_t>(imageCount)});
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(imageCount)});
    for(size_t i=0;i<MAX_LIGHT_SOURCE_COUNT;i++)
        poolSizes.push_back({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(imageCount)});
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(imageCount)});

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(imageCount);
    if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &second.DescriptorPool) != VK_SUCCESS)
    {throw std::runtime_error("failed to create descriptor pool!");}
}

void graphics::createSecondDescriptorSets(const std::vector<light<spotLight>*> & lightSource)
{
    std::vector<VkDescriptorSetLayout> layouts(imageCount, second.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = second.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();

    second.DescriptorSets.resize(imageCount);
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, second.DescriptorSets.data()) != VK_SUCCESS)
    {throw std::runtime_error("failed to allocate descriptor sets!");}

    for (size_t i = 0; i < imageCount; i++)
    {
        uint32_t index = 0;
        std::vector<VkWriteDescriptorSet> descriptorWrites(9);
        std::vector<VkDescriptorImageInfo> imageInfo(6);
        for(index = 0; index<6;index++)
        {
            imageInfo.at(index).imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.at(index).imageView = Attachments.at(3+index).imageView.at(i);
            imageInfo.at(index).sampler = VK_NULL_HANDLE;

            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = second.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pImageInfo = &imageInfo.at(index);
        }

        VkDescriptorBufferInfo lightBufferInfo;
            lightBufferInfo.buffer = second.lightUniformBuffers[i];
            lightBufferInfo.offset = 0;
            lightBufferInfo.range = sizeof(LightUniformBufferObject);

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
            bufferInfo.buffer = second.uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(SecondUniformBufferObject);

            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = second.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pBufferInfo = &lightBufferInfo;
        index++;
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = second.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.at(index).descriptorCount = MAX_LIGHT_SOURCE_COUNT;
            descriptorWrites.at(index).pImageInfo = shadowImageInfo;
        index++;
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = second.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    for(size_t i=0;i<lightSource.size();i++)
        lightSource[i]->createShadowDescriptorSets(second.lightUniformBuffers);
}

void graphics::Second::createUniformBuffers(VkApplication *app, uint32_t imageCount)
{
    lightUniformBuffers.resize(imageCount);
    lightUniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
    {
        createBuffer(app, sizeof(LightUniformBufferObject),
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     lightUniformBuffers[i], lightUniformBuffersMemory[i]);
    }

    uniformBuffers.resize(imageCount);
    uniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
    {
        createBuffer(app, sizeof(SecondUniformBufferObject),
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     uniformBuffers[i], uniformBuffersMemory[i]);
    }

    emptyUniformBuffers.resize(imageCount);
    emptyUniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
    {
        createBuffer(app, sizeof(LightBufferObject),
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     emptyUniformBuffers[i], emptyUniformBuffersMemory[i]);
    }
}

void graphics::Second::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i)
{
    vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &DescriptorSets[i], 0, nullptr);
    vkCmdDraw(commandBuffers[i], 6, 1, 0, 0);
}
