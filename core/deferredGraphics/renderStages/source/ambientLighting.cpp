#include "../graphics.h"
#include "../../../utils/operations.h"

#include <array>
#include <iostream>

struct lightPassPushConst{
    alignas(4) float                minAmbientFactor;
};

void deferredGraphics::AmbientLighting::DestroyPipeline(VkDevice* device){
    if(Pipeline)         {vkDestroyPipeline(*device, Pipeline, nullptr); Pipeline = VK_NULL_HANDLE;}
    if(PipelineLayout)   {vkDestroyPipelineLayout(*device, PipelineLayout, nullptr); PipelineLayout = VK_NULL_HANDLE;}
}

void deferredGraphics::AmbientLighting::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass){
    uint32_t index = 0;

    auto vertShaderCodeAmbient = ShaderModule::readFile(ExternalPath + "core\\deferredGraphics\\shaders\\ambientLightingPass\\ambientLightingVert.spv");
    auto fragShaderCodeAmbient = ShaderModule::readFile(ExternalPath + "core\\deferredGraphics\\shaders\\ambientLightingPass\\ambientLightingFrag.spv");
    VkShaderModule vertShaderModuleAmbient = ShaderModule::create(device, vertShaderCodeAmbient);
    VkShaderModule fragShaderModuleAmbient = ShaderModule::create(device, fragShaderCodeAmbient);
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
    std::array<VkDescriptorSetLayout,1> ambientSetLayouts = {Parent->DescriptorSetLayout};
    VkPipelineLayoutCreateInfo ambientPipelineLayoutInfo{};
        ambientPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ambientPipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(ambientSetLayouts.size());
        ambientPipelineLayoutInfo.pSetLayouts = ambientSetLayouts.data();
        ambientPipelineLayoutInfo.pushConstantRangeCount = 1;
        ambientPipelineLayoutInfo.pPushConstantRanges = ambientPushConstantRange.data();
    vkCreatePipelineLayout(*device, &ambientPipelineLayoutInfo, nullptr, &PipelineLayout);

    index = 0;
    std::array<VkGraphicsPipelineCreateInfo,1> pipelineInfoAmbient{};
        pipelineInfoAmbient[index].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfoAmbient[index].pNext = nullptr;
        pipelineInfoAmbient[index].stageCount = static_cast<uint32_t>(shaderStagesAmbient.size());
        pipelineInfoAmbient[index].pStages = shaderStagesAmbient.data();
        pipelineInfoAmbient[index].pVertexInputState = &vertexInputInfo;
        pipelineInfoAmbient[index].pInputAssemblyState = &inputAssembly;
        pipelineInfoAmbient[index].pViewportState = &viewportState;
        pipelineInfoAmbient[index].pRasterizationState = &rasterizer;
        pipelineInfoAmbient[index].pMultisampleState = &multisampling;
        pipelineInfoAmbient[index].pColorBlendState = &colorBlending;
        pipelineInfoAmbient[index].layout = PipelineLayout;
        pipelineInfoAmbient[index].renderPass = *pRenderPass;
        pipelineInfoAmbient[index].subpass = 1;
        pipelineInfoAmbient[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfoAmbient[index].pDepthStencilState = &AmbientDepthStencil;
    vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfoAmbient.size()), pipelineInfoAmbient.data(), nullptr, &Pipeline);

    vkDestroyShaderModule(*device, fragShaderModuleAmbient, nullptr);
    vkDestroyShaderModule(*device, vertShaderModuleAmbient, nullptr);
}

void deferredGraphics::AmbientLighting::render(uint32_t frameNumber, VkCommandBuffer commandBuffers){
    lightPassPushConst pushConst{};
        pushConst.minAmbientFactor = minAmbientFactor;
    vkCmdPushConstants(commandBuffers, PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(lightPassPushConst), &pushConst);

    vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    vkCmdBindDescriptorSets(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &Parent->DescriptorSets[frameNumber], 0, nullptr);
    vkCmdDraw(commandBuffers, 6, 1, 0, 0);
}
