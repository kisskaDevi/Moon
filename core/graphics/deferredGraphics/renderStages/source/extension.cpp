#include "../graphics.h"
#include "core/operations.h"
#include "core/transformational/object.h"
#include "core/transformational/gltfmodel.h"

#include <array>
#include <iostream>

struct OutliningPushConst{
    alignas(16) glm::vec4           stencilColor;
    alignas(4)  float               width;
};

void deferredGraphics::OutliningExtension::DestroyOutliningPipeline(VkDevice* device)
{
    if(outliningPipeline)         {vkDestroyPipeline(*device, outliningPipeline, nullptr); outliningPipeline = VK_NULL_HANDLE;}
    if(outliningPipelineLayout)   {vkDestroyPipelineLayout(*device, outliningPipelineLayout, nullptr); outliningPipelineLayout = VK_NULL_HANDLE;}
}

void deferredGraphics::OutliningExtension::createOutliningPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\outlining\\outliningVert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\outlining\\outliningFrag.spv");

    VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);
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
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
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
        pushConstantRange[index].size = sizeof(OutliningPushConst);
    std::array<VkDescriptorSetLayout,4> SetLayouts = {Parent->SceneDescriptorSetLayout,Parent->ObjectDescriptorSetLayout,Parent->PrimitiveDescriptorSetLayout,Parent->MaterialDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &outliningPipelineLayout);

    index=0;
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
        pipelineInfo[index].layout = outliningPipelineLayout;
        pipelineInfo[index].renderPass = *pRenderPass;
        pipelineInfo[index].subpass = 0;
        pipelineInfo[index].pDepthStencilState = &depthStencil;
        pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &outliningPipeline);

    vkDestroyShaderModule(*device, fragShaderModule, nullptr);
    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
}

void deferredGraphics::OutliningExtension::render(uint32_t frameNumber, VkCommandBuffer commandBuffers)
{
    vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, outliningPipeline);
    for(auto object: Parent->objects)
    {
        if(object->getEnable()&&object->getOutliningEnable()){
            VkDeviceSize offsets[1] = { 0 };
            vkCmdBindVertexBuffers(commandBuffers, 0, 1, & object->getModel(frameNumber)->vertices.buffer, offsets);
            if (object->getModel(frameNumber)->indices.buffer != VK_NULL_HANDLE)
                vkCmdBindIndexBuffer(commandBuffers, object->getModel(frameNumber)->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

            OutliningPushConst pushConst{};
                pushConst.stencilColor = object->getOutliningColor();
                pushConst.width = object->getOutliningWidth();
            vkCmdPushConstants(commandBuffers, outliningPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(OutliningPushConst), &pushConst);

            for (auto node : object->getModel(frameNumber)->nodes){
                std::vector<VkDescriptorSet> descriptorSets = {Parent->DescriptorSets[frameNumber],object->getDescriptorSet()[frameNumber]};
                renderNode(commandBuffers,node,&outliningPipelineLayout,static_cast<uint32_t>(descriptorSets.size()),descriptorSets.data(),nullptr);
            }
        }
    }
}

void deferredGraphics::OutliningExtension::renderNode(VkCommandBuffer commandBuffer, Node *node, VkPipelineLayout* pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t* primitiveCount)
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

            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, *pipelineLayout, 0, descriptorSetsCount+2, nodeDescriptorSets.data(), 0, NULL);

            if (primitive->hasIndices){
                vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
            }else{
                vkCmdDraw(commandBuffer, primitive->vertexCount, 1, 0, 0);
            }
        }
    }
    for (auto child : node->children)
        renderNode(commandBuffer,child,pipelineLayout,descriptorSetsCount,descriptorSets,primitiveCount);
}
