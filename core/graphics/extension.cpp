#include "graphics.h"
#include "core/operations.h"
#include "core/transformational/object.h"
#include "core/transformational/gltfmodel.h"

void graphics::bloomExtension::Destroy(VkApplication *app)
{
    vkDestroyPipeline(app->getDevice(), Pipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), PipelineLayout, nullptr);
}

void graphics::godRaysExtension::Destroy(VkApplication *app)
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

void graphics::bloomExtension::createPipeline(VkApplication *app, struct Base *base, graphicsInfo info)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\bloomSprite\\vertBloomSprite.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\bloomSprite\\fragBloomSprite.spv");
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

    auto bindingDescription = gltfModel::Vertex::getBloomBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getBloomAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
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
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
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
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout!");

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

    index = 0;
    std::array<VkGraphicsPipelineCreateInfo,1> pipelineInfo{};
        pipelineInfo[index].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo[index].stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo[index].pStages = shaderStages.data();
        pipelineInfo[index].pVertexInputState = &vertexInputInfo;
        pipelineInfo[index].pInputAssemblyState = &inputAssembly;
        pipelineInfo[index].pViewportState = &viewportState;
        pipelineInfo[index].pRasterizationState = &rasterizer;
        pipelineInfo[index].pMultisampleState = &multisampling;
        pipelineInfo[index].pColorBlendState = &colorBlending;
        pipelineInfo[index].layout = PipelineLayout;
        pipelineInfo[index].renderPass = info.renderPass;
        pipelineInfo[index].subpass = 0;
        pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo[index].pDepthStencilState = &depthStencil;
    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics pipeline!");

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::godRaysExtension::createPipeline(VkApplication *app, struct Base *base, graphicsInfo info)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\godRays\\godRaysVert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\godRays\\godRaysFrag.spv");
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

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
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
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
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
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout!");

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

    index = 0;
    std::array<VkGraphicsPipelineCreateInfo,1> pipelineInfo{};
        pipelineInfo[index].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo[index].stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo[index].pStages = shaderStages.data();
        pipelineInfo[index].pVertexInputState = &vertexInputInfo;
        pipelineInfo[index].pInputAssemblyState = &inputAssembly;
        pipelineInfo[index].pViewportState = &viewportState;
        pipelineInfo[index].pRasterizationState = &rasterizer;
        pipelineInfo[index].pMultisampleState = &multisampling;
        pipelineInfo[index].pColorBlendState = &colorBlending;
        pipelineInfo[index].layout = PipelineLayout;
        pipelineInfo[index].renderPass = info.renderPass;
        pipelineInfo[index].subpass = 0;
        pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo[index].pDepthStencilState = &depthStencil;
    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics pipeline!");

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::StencilExtension::createFirstPipeline(VkApplication *app, Base *base, graphicsInfo info)
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
        throw std::runtime_error("failed to create pipeline layout!");

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
        throw std::runtime_error("failed to create graphics pipeline!");

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::StencilExtension::createSecondPipeline(VkApplication *app, Base *base, graphicsInfo info)
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
        throw std::runtime_error("failed to create pipeline layout!");

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
        throw std::runtime_error("failed to create graphics pipeline!");

    //можно удалить шейдерные модули после использования
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::bloomExtension::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i, graphics *Graphics, Base *base)
{
    vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    for(size_t j = 0; j<objects.size() ;j++)
    {
        VkDeviceSize offsets[1] = { 0 };
        vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, & objects[j]->getModel()->vertices.buffer, offsets);
        if (objects[j]->getModel()->indices.buffer != VK_NULL_HANDLE)
            vkCmdBindIndexBuffer(commandBuffers[i],  objects[j]->getModel()->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

        for (auto node : objects[j]->getModel()->nodes)
            Graphics->renderNode(node,commandBuffers[i],base->DescriptorSets[i],objects[j]->getDescriptorSet()[i],PipelineLayout);
    }
}
void graphics::bloomExtension::setMaterials(std::vector<PushConstBlockMaterial> &nodeMaterials, graphics *Graphics)
{
    for(size_t j = 0; j<objects.size() ;j++)
        for (auto node : objects[j]->getModel()->nodes)
            Graphics->setMaterialNode(node,nodeMaterials);
}

void graphics::StencilExtension::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i, graphics *Graphics, Base *base)
{
    vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, firstPipeline);
    for(size_t j = 0; j<objects.size() ;j++)
    {
        VkDeviceSize offsets[1] = { 0 };
        vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, & objects[j]->getModel()->vertices.buffer, offsets);
        if (objects[j]->getModel()->indices.buffer != VK_NULL_HANDLE)
            vkCmdBindIndexBuffer(commandBuffers[i], objects[j]->getModel()->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

        for (auto node : objects[j]->getModel()->nodes)
            Graphics->renderNode(node,commandBuffers[i],base->DescriptorSets[i],objects[j]->getDescriptorSet()[i],firstPipelineLayout);
    }

    for(size_t j = 0; j<objects.size() ;j++)
    {
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
                Graphics->stencilRenderNode(node,commandBuffers[i],base->DescriptorSets[i],objects[j]->getDescriptorSet()[i],secondPipelineLayout);
        }
    }
}
void graphics::StencilExtension::setMaterials(std::vector<PushConstBlockMaterial> &nodeMaterials, graphics *Graphics)
{
    for(size_t j = 0; j<objects.size() ;j++)
        for (auto node : objects[j]->getModel()->nodes)
            Graphics->setMaterialNode(node,nodeMaterials);
}
