#include "../graphics.h"
#include "core/operations.h"
#include "../../bufferObjects.h"

#include <array>
#include <iostream>

void deferredGraphics::Lighting::createAmbientPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    uint32_t index = 0;

    auto vertShaderCodeAmbient = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\ambientLightingPass\\ambientLightingVert.spv");
    auto fragShaderCodeAmbient = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\ambientLightingPass\\ambientLightingFrag.spv");
    VkShaderModule vertShaderModuleAmbient = createShaderModule(device, vertShaderCodeAmbient);
    VkShaderModule fragShaderModuleAmbient = createShaderModule(device, fragShaderCodeAmbient);
    std::array<VkPipelineShaderStageCreateInfo,2> shaderStagesAmbient{};
        shaderStagesAmbient[index].pName = "main";
        shaderStagesAmbient[index].module = fragShaderModuleAmbient;
        shaderStagesAmbient[index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStagesAmbient[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    index++;
        shaderStagesAmbient[index].pName = "main";
        shaderStagesAmbient[index].module = vertShaderModuleAmbient;
        shaderStagesAmbient[index].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStagesAmbient[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

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
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = pInfo->Samples;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

    std::array<VkPipelineColorBlendAttachmentState,3> colorBlendAttachment;
    for(uint32_t index=0;index<colorBlendAttachment.size();index++)
    {
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_TRUE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
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

    VkPipelineDepthStencilStateCreateInfo AmbientDepthStencil{};
        AmbientDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        AmbientDepthStencil.depthTestEnable = VK_FALSE;
        AmbientDepthStencil.depthWriteEnable = VK_FALSE;
        AmbientDepthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        AmbientDepthStencil.depthBoundsTestEnable = VK_FALSE;
        AmbientDepthStencil.minDepthBounds = 0.0f;
        AmbientDepthStencil.maxDepthBounds = 1.0f;
        AmbientDepthStencil.stencilTestEnable = VK_FALSE;
        AmbientDepthStencil.front = {};
        AmbientDepthStencil.back = {};

    index = 0;
    std::array<VkPushConstantRange,1> ambientPushConstantRange{};
        ambientPushConstantRange[index].stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        ambientPushConstantRange[index].offset = 0;
        ambientPushConstantRange[index].size = sizeof(lightPassPushConst);
    std::array<VkDescriptorSetLayout,1> ambientSetLayouts = {DescriptorSetLayout};
    VkPipelineLayoutCreateInfo ambientPipelineLayoutInfo{};
        ambientPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ambientPipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(ambientSetLayouts.size());
        ambientPipelineLayoutInfo.pSetLayouts = ambientSetLayouts.data();
        ambientPipelineLayoutInfo.pushConstantRangeCount = 1;
        ambientPipelineLayoutInfo.pPushConstantRanges = ambientPushConstantRange.data();
    if (vkCreatePipelineLayout(*device, &ambientPipelineLayoutInfo, nullptr, &AmbientPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass pipeline layout!");

    index = 0;
    std::array<VkGraphicsPipelineCreateInfo,1> pipelineInfoAmbient{};
        pipelineInfoAmbient[index].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfoAmbient[index].stageCount = static_cast<uint32_t>(shaderStagesAmbient.size());
        pipelineInfoAmbient[index].pStages = shaderStagesAmbient.data();
        pipelineInfoAmbient[index].pVertexInputState = &vertexInputInfo;
        pipelineInfoAmbient[index].pInputAssemblyState = &inputAssembly;
        pipelineInfoAmbient[index].pViewportState = &viewportState;
        pipelineInfoAmbient[index].pRasterizationState = &rasterizer;
        pipelineInfoAmbient[index].pMultisampleState = &multisampling;
        pipelineInfoAmbient[index].pColorBlendState = &colorBlending;
        pipelineInfoAmbient[index].layout = AmbientPipelineLayout;
        pipelineInfoAmbient[index].renderPass = *pRenderPass;
        pipelineInfoAmbient[index].subpass = 1;
        pipelineInfoAmbient[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfoAmbient[index].pDepthStencilState = &AmbientDepthStencil;
    if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfoAmbient.size()), pipelineInfoAmbient.data(), nullptr, &AmbientPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass graphics pipeline!");

    vkDestroyShaderModule(*device, fragShaderModuleAmbient, nullptr);
    vkDestroyShaderModule(*device, vertShaderModuleAmbient, nullptr);
}

void deferredGraphics::Lighting::createSpotPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    VkPipeline SpotPipeline;
    VkPipelineLayout SpotPipelineLayout;
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\spotLightingVert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\spotLightingFrag.spv");
    VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);
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

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

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
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = pInfo->Samples;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

    std::array<VkPipelineColorBlendAttachmentState,3> colorBlendAttachment;
    for(uint32_t index=0;index<colorBlendAttachment.size();index++)
    {
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_TRUE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
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
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};

    index = 0;
    std::array<VkPushConstantRange,1> pushConstantRange{};
        pushConstantRange[index].stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange[index].offset = 0;
        pushConstantRange[index].size = sizeof(lightPassPushConst);
    std::array<VkDescriptorSetLayout,2> SetLayouts = {DescriptorSetLayout,LightDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    if (vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &SpotPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass pipeline layout!");

    index = 0;
    std::array<VkGraphicsPipelineCreateInfo,1> pipelineInfoScattering{};
        pipelineInfoScattering[index].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfoScattering[index].stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfoScattering[index].pStages = shaderStages.data();
        pipelineInfoScattering[index].pVertexInputState = &vertexInputInfo;
        pipelineInfoScattering[index].pInputAssemblyState = &inputAssembly;
        pipelineInfoScattering[index].pViewportState = &viewportState;
        pipelineInfoScattering[index].pRasterizationState = &rasterizer;
        pipelineInfoScattering[index].pMultisampleState = &multisampling;
        pipelineInfoScattering[index].pColorBlendState = &colorBlending;
        pipelineInfoScattering[index].layout = SpotPipelineLayout;
        pipelineInfoScattering[index].renderPass = *pRenderPass;
        pipelineInfoScattering[index].subpass = 1;
        pipelineInfoScattering[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfoScattering[index].pDepthStencilState = &depthStencil;
    if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfoScattering.size()), pipelineInfoScattering.data(), nullptr, &SpotPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass graphics pipeline!");

    PipelineLayoutDictionary[(false<<5)|(false <<4)|(0x0)] = SpotPipelineLayout;
    PipelinesDictionary[(false<<5)|(false <<4)|(0x0)] = SpotPipeline;

    vkDestroyShaderModule(*device, fragShaderModule, nullptr);
    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
}

void deferredGraphics::Lighting::createShadowSpotPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    VkPipelineLayout ShadowSpotPipelineLayout;
    VkPipeline ShadowSpotPipeline;

    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\spotLightingVert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\shadowSpotLightingFrag.spv");
    VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);
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

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

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
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = pInfo->Samples;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

    std::array<VkPipelineColorBlendAttachmentState,3> colorBlendAttachment;
    for(uint32_t index=0;index<colorBlendAttachment.size();index++)
    {
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_TRUE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
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
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};

    index = 0;
    std::array<VkPushConstantRange,1> pushConstantRange{};
        pushConstantRange[index].stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange[index].offset = 0;
        pushConstantRange[index].size = sizeof(lightPassPushConst);
    std::array<VkDescriptorSetLayout,2> SetLayouts = {DescriptorSetLayout,LightDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    if (vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &ShadowSpotPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass pipeline layout!");

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
        pipelineInfo[index].layout = ShadowSpotPipelineLayout;
        pipelineInfo[index].renderPass = *pRenderPass;
        pipelineInfo[index].subpass = 1;
        pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo[index].pDepthStencilState = &depthStencil;
    if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &ShadowSpotPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass graphics pipeline!");

    PipelineLayoutDictionary[(false<<5)|(true <<4)|(0x0)] = ShadowSpotPipelineLayout;
    PipelinesDictionary[(false<<5)|(true <<4)|(0x0)] = ShadowSpotPipeline;

    vkDestroyShaderModule(*device, fragShaderModule, nullptr);
    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
}

void deferredGraphics::Lighting::createScatteringSpotPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    VkPipelineLayout ScatteringSpotPipelineLayout;
    VkPipeline ScatteringSpotPipeline;

    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\spotLightingVert.spv");
    auto fragShaderCode = enableScattering ?
                readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\scatteringSpotLightingFrag.spv") :
                readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\spotLightingFrag.spv") ;
    VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);
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

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

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
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = pInfo->Samples;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

    std::array<VkPipelineColorBlendAttachmentState,3> colorBlendAttachment;
    for(uint32_t index=0;index<colorBlendAttachment.size();index++)
    {
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_TRUE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
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
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};

    index = 0;
    std::array<VkPushConstantRange,1> pushConstantRange{};
        pushConstantRange[index].stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange[index].offset = 0;
        pushConstantRange[index].size = sizeof(lightPassPushConst);
    std::array<VkDescriptorSetLayout,2> SetLayouts = {DescriptorSetLayout,LightDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    if (vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &ScatteringSpotPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass pipeline layout!");

    index = 0;
    std::array<VkGraphicsPipelineCreateInfo,1> pipelineInfoScattering{};
        pipelineInfoScattering[index].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfoScattering[index].stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfoScattering[index].pStages = shaderStages.data();
        pipelineInfoScattering[index].pVertexInputState = &vertexInputInfo;
        pipelineInfoScattering[index].pInputAssemblyState = &inputAssembly;
        pipelineInfoScattering[index].pViewportState = &viewportState;
        pipelineInfoScattering[index].pRasterizationState = &rasterizer;
        pipelineInfoScattering[index].pMultisampleState = &multisampling;
        pipelineInfoScattering[index].pColorBlendState = &colorBlending;
        pipelineInfoScattering[index].layout = ScatteringSpotPipelineLayout;
        pipelineInfoScattering[index].renderPass = *pRenderPass;
        pipelineInfoScattering[index].subpass = 1;
        pipelineInfoScattering[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfoScattering[index].pDepthStencilState = &depthStencil;
    if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfoScattering.size()), pipelineInfoScattering.data(), nullptr, &ScatteringSpotPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass graphics pipeline!");

    PipelineLayoutDictionary[(true <<5)|(false <<4)|(0x0)] = ScatteringSpotPipelineLayout;
    PipelinesDictionary[(true <<5)|(false <<4)|(0x0)] = ScatteringSpotPipeline;

    vkDestroyShaderModule(*device, fragShaderModule, nullptr);
    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
}

void deferredGraphics::Lighting::createScatteringShadowSpotPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    VkPipelineLayout ScatteringShadowSpotPipelineLayout;
    VkPipeline ScatteringShadowSpotPipeline;

    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\spotLightingVert.spv");
    auto fragShaderCode = enableScattering ?
                readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\scatteringShadowSpotLightingFrag.spv"):
                readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\shadowSpotLightingFrag.spv");
    VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);
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

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

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
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = pInfo->Samples;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

    std::array<VkPipelineColorBlendAttachmentState,3> colorBlendAttachment;
    for(uint32_t index=0;index<colorBlendAttachment.size();index++)
    {
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_TRUE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
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
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};

    index = 0;
    std::array<VkPushConstantRange,1> pushConstantRange{};
        pushConstantRange[index].stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange[index].offset = 0;
        pushConstantRange[index].size = sizeof(lightPassPushConst);
    std::array<VkDescriptorSetLayout,2> SetLayouts = {DescriptorSetLayout,LightDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    if (vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &ScatteringShadowSpotPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass pipeline layout!");

    index = 0;
    std::array<VkGraphicsPipelineCreateInfo,1> pipelineInfoScattering{};
        pipelineInfoScattering[index].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfoScattering[index].stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfoScattering[index].pStages = shaderStages.data();
        pipelineInfoScattering[index].pVertexInputState = &vertexInputInfo;
        pipelineInfoScattering[index].pInputAssemblyState = &inputAssembly;
        pipelineInfoScattering[index].pViewportState = &viewportState;
        pipelineInfoScattering[index].pRasterizationState = &rasterizer;
        pipelineInfoScattering[index].pMultisampleState = &multisampling;
        pipelineInfoScattering[index].pColorBlendState = &colorBlending;
        pipelineInfoScattering[index].layout = ScatteringShadowSpotPipelineLayout;
        pipelineInfoScattering[index].renderPass = *pRenderPass;
        pipelineInfoScattering[index].subpass = 1;
        pipelineInfoScattering[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfoScattering[index].pDepthStencilState = &depthStencil;
    if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfoScattering.size()), pipelineInfoScattering.data(), nullptr, &ScatteringShadowSpotPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass graphics pipeline!");

    PipelineLayoutDictionary[(true <<5)|(true <<4)|(0x0)] = ScatteringShadowSpotPipelineLayout;
    PipelinesDictionary[(true <<5)|(true <<4)|(0x0)] = ScatteringShadowSpotPipeline;

    vkDestroyShaderModule(*device, fragShaderModule, nullptr);
    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
}
