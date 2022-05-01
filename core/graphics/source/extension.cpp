#include "core/graphics/graphics.h"
#include "core/operations.h"
#include "core/transformational/object.h"
#include "core/transformational/gltfmodel.h"

#include <array>

void graphics::bloomExtension::Destroy(VkApplication *app)
{
    vkDestroyPipeline(app->getDevice(), Pipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), PipelineLayout, nullptr);
}

void graphics::oneColorExtension::Destroy(VkApplication *app)
{
    vkDestroyPipeline(app->getDevice(), Pipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), PipelineLayout,nullptr);
}

void graphics::StencilExtension::DestroyFirstPipeline(VkApplication *app)
{
    vkDestroyPipeline(app->getDevice(), firstPipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), firstPipelineLayout,nullptr);
}

void graphics::StencilExtension::DestroySecondPipeline(VkApplication *app)
{
    vkDestroyPipeline(app->getDevice(), secondPipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), secondPipelineLayout,nullptr);
}

void graphics::bloomExtension::createPipeline(VkApplication *app, graphicsInfo info)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\bloom\\vertBloom.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\bloom\\fragBloom.spv");
    VkShaderModule vertShaderModule = createShaderModule(app, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(app, fragShaderCode);
    std::array<VkPipelineShaderStageCreateInfo,2> shaderStages{};
        shaderStages[index].pName = "main";
        shaderStages[index].module = fragShaderModule;
        shaderStages[index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    index++;
        shaderStages[index].pName = "main";
        shaderStages[index].module = vertShaderModule;
        shaderStages[index].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

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
            viewport[index].width = (float) info.extent.width;
            viewport[index].height = (float) info.extent.height;
            viewport[index].minDepth = 0.0f;
            viewport[index].maxDepth = 1.0f;
        std::array<VkRect2D,1> scissor{};
            scissor[index].offset = {0, 0};
            scissor[index].extent = info.extent;
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
            multisampling.rasterizationSamples = info.msaaSamples;
            multisampling.minSampleShading = 1.0f;
            multisampling.pSampleMask = nullptr;
            multisampling.alphaToCoverageEnable = VK_FALSE;
            multisampling.alphaToOneEnable = VK_FALSE;

        /* Последней стадией в графическом конвейере является стадия смешивания цветов. Эта стадия отвечает за запись фрагментов
         * в цветовые подключения. Во многих случаях это простая операция, которая просто записывает содержимое выходного значения
         * фрагментного шейдера поверх старого значения. Однакоподдеживаются смешивание этих значнеий со значениями,
         * уже находящимися во фрейм буфере, и выполнение простых логических операций между выходными значениями фрагментного
         * шейдера и текущим содержанием фреймбуфера.*/

        std::array<VkPipelineColorBlendAttachmentState,6> colorBlendAttachment;
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
            pushConstantRange[index].size = sizeof(PushConst);
        std::array<VkDescriptorSetLayout,4> setLayouts = {base->SceneDescriptorSetLayout,base->ObjectDescriptorSetLayout,base->PrimitiveDescriptorSetLayout,base->MaterialDescriptorSetLayout};
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
            pipelineLayoutInfo.pSetLayouts = setLayouts.data();
            pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
            pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
        if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
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
            pipelineInfo[index].renderPass = info.renderPass;                              //проход рендеринга
            pipelineInfo[index].subpass = 0;                                               //подпроход рендеригка
            pipelineInfo[index].pDepthStencilState = &depthStencil;
            pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
        if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create base graphics pipeline!");

        //можно удалить шейдерные модули после использования
        vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
        vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::oneColorExtension::createPipeline(VkApplication *app, graphicsInfo info)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\oneColor\\oneColorVert.spv");   //считываем шейдеры
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\oneColor\\oneColorFrag.spv");
    VkShaderModule vertShaderModule = createShaderModule(app, vertShaderCode);                      //создаём шейдерные модули
    VkShaderModule fragShaderModule = createShaderModule(app, fragShaderCode);
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
        viewport[index].width = (float) info.extent.width;
        viewport[index].height = (float) info.extent.height;
        viewport[index].minDepth = 0.0f;
        viewport[index].maxDepth = 1.0f;
    std::array<VkRect2D,1> scissor{};
        scissor[index].offset = {0, 0};
        scissor[index].extent = info.extent;
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
        multisampling.rasterizationSamples = info.msaaSamples;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

    /* Последней стадией в графическом конвейере является стадия смешивания цветов. Эта стадия отвечает за запись фрагментов
     * в цветовые подключения. Во многих случаях это простая операция, которая просто записывает содержимое выходного значения
     * фрагментного шейдера поверх старого значения. Однакоподдеживаются смешивание этих значнеий со значениями,
     * уже находящимися во фрейм буфере, и выполнение простых логических операций между выходными значениями фрагментного
     * шейдера и текущим содержанием фреймбуфера.*/

    std::array<VkPipelineColorBlendAttachmentState,6> colorBlendAttachment;
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
        pushConstantRange[index].size = sizeof(PushConst);
    std::array<VkDescriptorSetLayout,4> setLayouts = {base->SceneDescriptorSetLayout,base->ObjectDescriptorSetLayout,base->PrimitiveDescriptorSetLayout,base->MaterialDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
        pipelineLayoutInfo.pSetLayouts = setLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
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
        pipelineInfo[index].renderPass = info.renderPass;                              //проход рендеринга
        pipelineInfo[index].subpass = 0;                                               //подпроход рендеригка
        pipelineInfo[index].pDepthStencilState = &depthStencil;
        pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create base graphics pipeline!");

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::StencilExtension::createFirstPipeline(VkApplication *app, graphicsInfo info)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\stencil\\firststencilvert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\stencil\\firststencilfrag.spv");
    //создаём шейдерные модули
    VkShaderModule vertShaderModule = createShaderModule(app, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(app, fragShaderCode);
    std::array<VkPipelineShaderStageCreateInfo,2> shaderStages{};
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[index].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[index].module = vertShaderModule;
        shaderStages[index].pName = "main";
    index++;
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[index].module = fragShaderModule;
        shaderStages[index].pName = "main";

    auto bindingDescription = gltfModel::Vertex::getStencilBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getStencilAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

    index = 0;
    std::array<VkViewport,1> viewport{};
        viewport[index].x = 0.0f;
        viewport[index].y = 0.0f;
        viewport[index].width = (float) info.extent.width;
        viewport[index].height = (float) info.extent.height;
        viewport[index].minDepth = 0.0f;
        viewport[index].maxDepth = 1.0f;
    std::array<VkRect2D,1> scissor{};
        scissor[index].offset = {0, 0};
        scissor[index].extent = info.extent;
    VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = static_cast<uint32_t>(viewport.size());
        viewportState.pViewports = viewport.data();
        viewportState.scissorCount = static_cast<uint32_t>(scissor.size());
        viewportState.pScissors = scissor.data();

    VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = info.msaaSamples;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

    std::array<VkPipelineColorBlendAttachmentState,6> colorBlendAttachment;
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
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachment.size());
        colorBlending.pAttachments = colorBlendAttachment.data();
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_TRUE;
        depthStencil.back.compareOp = VK_COMPARE_OP_ALWAYS;
        depthStencil.back.failOp = VK_STENCIL_OP_REPLACE;
        depthStencil.back.depthFailOp = VK_STENCIL_OP_REPLACE;
        depthStencil.back.passOp = VK_STENCIL_OP_REPLACE;
        depthStencil.back.compareMask = 0xff;
        depthStencil.back.writeMask = 0xff;
        depthStencil.back.reference = 1;
        depthStencil.front = depthStencil.back;

    index = 0;
    std::array<VkPushConstantRange,1> pushConstantRange;
        pushConstantRange[index].stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange[index].offset = 0;
        pushConstantRange[index].size = sizeof(PushConst);
    std::array<VkDescriptorSetLayout,4> SetLayouts = {base->SceneDescriptorSetLayout,base->ObjectDescriptorSetLayout,base->PrimitiveDescriptorSetLayout,base->MaterialDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &firstPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create stencil extension pipeline layout!");

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
        pipelineInfo[index].layout = firstPipelineLayout;
        pipelineInfo[index].renderPass = info.renderPass;                              //проход рендеринга
        pipelineInfo[index].subpass = 0;                                               //подпроход рендеригка
        pipelineInfo[index].pDepthStencilState = &depthStencil;
        pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &firstPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create stencil extension graphics pipeline!");

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::StencilExtension::createSecondPipeline(VkApplication *app, graphicsInfo info)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\stencil\\secondstencilvert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\stencil\\secondstencilfrag.spv");
    VkShaderModule vertShaderModule = createShaderModule(app, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(app, fragShaderCode);
    std::array<VkPipelineShaderStageCreateInfo,2> shaderStages{};
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[index].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[index].module = vertShaderModule;
        shaderStages[index].pName = "main";
    index++;
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[index].module = fragShaderModule;
        shaderStages[index].pName = "main";

    auto bindingDescription = gltfModel::Vertex::getStencilBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getStencilAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

    index = 0;
    std::array<VkViewport,1> viewport{};
        viewport[index].x = 0.0f;
        viewport[index].y = 0.0f;
        viewport[index].width = (float) info.extent.width;
        viewport[index].height = (float) info.extent.height;
        viewport[index].minDepth = 0.0f;
        viewport[index].maxDepth = 1.0f;
    std::array<VkRect2D,1> scissor{};
        scissor[index].offset = {0, 0};
        scissor[index].extent = info.extent;
    VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = static_cast<uint32_t>(viewport.size());
        viewportState.pViewports = viewport.data();
        viewportState.scissorCount = static_cast<uint32_t>(scissor.size());
        viewportState.pScissors = scissor.data();

    VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = info.msaaSamples;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

    std::array<VkPipelineColorBlendAttachmentState,6> colorBlendAttachment;
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
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachment.size());
        colorBlending.pAttachments = colorBlendAttachment.data();
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_TRUE;
        depthStencil.back.compareOp = VK_COMPARE_OP_NOT_EQUAL;
        depthStencil.back.failOp = VK_STENCIL_OP_KEEP;
        depthStencil.back.depthFailOp = VK_STENCIL_OP_KEEP;
        depthStencil.back.passOp = VK_STENCIL_OP_REPLACE;
        depthStencil.back.compareMask = 0xff;
        depthStencil.back.writeMask = 0xff;
        depthStencil.back.reference = 1;
        depthStencil.front = depthStencil.back;

    index = 0;
    std::array<VkPushConstantRange,1> pushConstantRange;
        pushConstantRange[index].stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange[index].offset = 0;
        pushConstantRange[index].size = sizeof(StencilPushConst);
    std::array<VkDescriptorSetLayout,4> SetLayouts = {base->SceneDescriptorSetLayout,base->ObjectDescriptorSetLayout,base->PrimitiveDescriptorSetLayout,base->MaterialDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &secondPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create second stencil extension pipeline layout!");

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
        pipelineInfo[index].layout = secondPipelineLayout;
        pipelineInfo[index].renderPass = info.renderPass;                              //проход рендеринга
        pipelineInfo[index].subpass = 0;                                               //подпроход рендеригка
        pipelineInfo[index].pDepthStencilState = &depthStencil;
        pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &secondPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create second stencil extension graphics pipeline!");

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::bloomExtension::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i, uint32_t& primitiveCount)
{
    vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    for(size_t j = 0; j<objects.size() ;j++)
    {
        if(objects[j]->getEnable()){
            VkDeviceSize offsets[1] = { 0 };
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, & objects[j]->getModel()->vertices.buffer, offsets);
            if (objects[j]->getModel()->indices.buffer != VK_NULL_HANDLE)
                vkCmdBindIndexBuffer(commandBuffers[i],  objects[j]->getModel()->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

            for (auto node : objects[j]->getModel()->nodes)
                renderNode(node,commandBuffers[i],base->DescriptorSets[i],objects[j]->getDescriptorSet()[i],PipelineLayout, primitiveCount);
        }
    }
}
void graphics::bloomExtension::setMaterials(std::vector<MaterialBlock> &nodeMaterials)
{
    for(size_t j = 0; j<objects.size() ;j++)
        for (auto node : objects[j]->getModel()->nodes){
            uint32_t objectPrimitive = 0;
            setMaterialNode(node,nodeMaterials,objectPrimitive,objects[j]->getModel()->firstPrimitive);
        }
}

void graphics::bloomExtension::renderNode(Node *node, VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, VkDescriptorSet& objectDescriptorSet, VkPipelineLayout& layout, uint32_t& primitiveCount)
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
            PushConst pushConst{};
                pushConst.normalTextureSet = primitive->material.normalTexture != nullptr ? primitive->material.texCoordSets.normal : -1;
                pushConst.number = primitiveCount;
            vkCmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConst), &pushConst);

            if (primitive->hasIndices)
                vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
            else
                vkCmdDraw(commandBuffer, primitive->vertexCount, 1, 0, 0);

            primitiveCount++;
        }
    }
    for (auto child : node->children)
        renderNode(child, commandBuffer, descriptorSet,objectDescriptorSet,layout, primitiveCount);
}

void graphics::bloomExtension::setMaterialNode(Node *node, std::vector<MaterialBlock> &nodeMaterials, uint32_t &objectPrimitive, const uint32_t firstPrimitive)
{
    if (node->mesh)
    {
        for (Primitive* primitive : node->mesh->primitives)
        {
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

            pushConstBlockMaterial.primitive = objectPrimitive;
            pushConstBlockMaterial.firstIndex = firstPrimitive;

            nodeMaterials.push_back(pushConstBlockMaterial);

            objectPrimitive++;
        }
    }
    for (auto child : node->children)
        setMaterialNode(child, nodeMaterials, objectPrimitive, firstPrimitive);
}

void graphics::oneColorExtension::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i, uint32_t& primitiveCount)
{
    vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    for(size_t j = 0; j<objects.size() ;j++)
    {
        if(objects[j]->getEnable()){
            VkDeviceSize offsets[1] = { 0 };
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, & objects[j]->getModel()->vertices.buffer, offsets);
            if (objects[j]->getModel()->indices.buffer != VK_NULL_HANDLE)
                vkCmdBindIndexBuffer(commandBuffers[i],  objects[j]->getModel()->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

            for (auto node : objects[j]->getModel()->nodes)
                renderNode(node,commandBuffers[i],base->DescriptorSets[i],objects[j]->getDescriptorSet()[i],PipelineLayout, primitiveCount);
        }
    }
}
void graphics::oneColorExtension::setMaterials(std::vector<MaterialBlock> &nodeMaterials)
{
    for(size_t j = 0; j<objects.size() ;j++)
        for (auto node : objects[j]->getModel()->nodes){
            uint32_t objectPrimitive = 0;
            setMaterialNode(node,nodeMaterials,objectPrimitive,objects[j]->getModel()->firstPrimitive);
        }
}

void graphics::oneColorExtension::renderNode(Node *node, VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, VkDescriptorSet& objectDescriptorSet, VkPipelineLayout& layout, uint32_t& primitiveCount)
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
            PushConst pushConst{};
                pushConst.normalTextureSet = primitive->material.normalTexture != nullptr ? primitive->material.texCoordSets.normal : -1;
                pushConst.number = primitiveCount;
            vkCmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConst), &pushConst);

            if (primitive->hasIndices)
                vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
            else
                vkCmdDraw(commandBuffer, primitive->vertexCount, 1, 0, 0);

            primitiveCount++;
        }
    }
    for (auto child : node->children)
        renderNode(child, commandBuffer, descriptorSet,objectDescriptorSet,layout, primitiveCount);
}

void graphics::oneColorExtension::setMaterialNode(Node *node, std::vector<MaterialBlock> &nodeMaterials, uint32_t &objectPrimitive, const uint32_t firstPrimitive)
{
    if (node->mesh)
    {
        for (Primitive* primitive : node->mesh->primitives)
        {
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

            pushConstBlockMaterial.primitive = objectPrimitive;
            pushConstBlockMaterial.firstIndex = firstPrimitive;

            nodeMaterials.push_back(pushConstBlockMaterial);

            objectPrimitive++;
        }
    }
    for (auto child : node->children)
        setMaterialNode(child, nodeMaterials, objectPrimitive, firstPrimitive);
}

void graphics::StencilExtension::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i, uint32_t& primitiveCount)
{
    vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, firstPipeline);
    for(size_t j = 0; j<objects.size() ;j++)
    {
        if(objects[j]->getEnable()){
            VkDeviceSize offsets[1] = { 0 };
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, & objects[j]->getModel()->vertices.buffer, offsets);
            if (objects[j]->getModel()->indices.buffer != VK_NULL_HANDLE)
                vkCmdBindIndexBuffer(commandBuffers[i], objects[j]->getModel()->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

            for (auto node : objects[j]->getModel()->nodes)
                renderNode(node,commandBuffers[i],base->DescriptorSets[i],objects[j]->getDescriptorSet()[i],firstPipelineLayout, primitiveCount);
        }
    }

    for(size_t j = 0; j<objects.size() ;j++)
    {
        if(objects[j]->getEnable()){
            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, secondPipeline);
            if(stencilEnable[j]){
                VkDeviceSize offsets[1] = { 0 };
                vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, & objects[j]->getModel()->vertices.buffer, offsets);
                if (objects[j]->getModel()->indices.buffer != VK_NULL_HANDLE)
                    vkCmdBindIndexBuffer(commandBuffers[i], objects[j]->getModel()->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

                StencilPushConst pushConst{};
                    pushConst.stencilColor = stencilColor[j];
                vkCmdPushConstants(commandBuffers[i], secondPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(StencilPushConst), &pushConst);

                for (auto node : objects[j]->getModel()->nodes)
                    stencilRenderNode(node,commandBuffers[i],base->DescriptorSets[i],objects[j]->getDescriptorSet()[i],secondPipelineLayout);
            }
        }
    }
}
void graphics::StencilExtension::setMaterials(std::vector<MaterialBlock> &nodeMaterials)
{
    for(size_t j = 0; j<objects.size() ;j++)
        for (auto node : objects[j]->getModel()->nodes){
            uint32_t objectPrimitive = 0;
            setMaterialNode(node,nodeMaterials,objectPrimitive,objects[j]->getModel()->firstPrimitive);
        }
}

void graphics::StencilExtension::renderNode(Node *node, VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, VkDescriptorSet& objectDescriptorSet, VkPipelineLayout& layout, uint32_t& primitiveCount)
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
            PushConst pushConst{};
                pushConst.normalTextureSet = primitive->material.normalTexture != nullptr ? primitive->material.texCoordSets.normal : -1;
                pushConst.number = primitiveCount;
            vkCmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConst), &pushConst);

            if (primitive->hasIndices)
                vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
            else
                vkCmdDraw(commandBuffer, primitive->vertexCount, 1, 0, 0);

            primitiveCount++;
        }
    }
    for (auto child : node->children)
        renderNode(child, commandBuffer, descriptorSet,objectDescriptorSet,layout, primitiveCount);
}

void graphics::StencilExtension::stencilRenderNode(Node* node, VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, VkDescriptorSet& objectDescriptorSet, VkPipelineLayout& layout)
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

            if (primitive->hasIndices)
                vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
            else
                vkCmdDraw(commandBuffer, primitive->vertexCount, 1, 0, 0);
        }
    }
    for (auto child : node->children)
        stencilRenderNode(child, commandBuffer, descriptorSet,objectDescriptorSet,layout);
}

void graphics::StencilExtension::setMaterialNode(Node *node, std::vector<MaterialBlock> &nodeMaterials, uint32_t &objectPrimitive, const uint32_t firstPrimitive)
{
    if (node->mesh)
    {
        for (Primitive* primitive : node->mesh->primitives)
        {
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

            pushConstBlockMaterial.primitive = objectPrimitive;
            pushConstBlockMaterial.firstIndex = firstPrimitive;

            nodeMaterials.push_back(pushConstBlockMaterial);

            objectPrimitive++;
        }
    }
    for (auto child : node->children)
        setMaterialNode(child, nodeMaterials, objectPrimitive, firstPrimitive);
}
