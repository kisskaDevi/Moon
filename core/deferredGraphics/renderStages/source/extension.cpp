#include "../graphics.h"
#include "../../../utils/operations.h"
#include "../../utils/vkdefault.h"
#include "../../../transformational/object.h"
#include "../../../models/gltfmodel.h"

struct OutliningPushConst{
    alignas(16) glm::vec4           stencilColor;
    alignas(4)  float               width;
};

void graphics::OutliningExtension::DestroyPipeline(VkDevice device)
{
    if(Pipeline)         {vkDestroyPipeline(device, Pipeline, nullptr); Pipeline = VK_NULL_HANDLE;}
    if(PipelineLayout)   {vkDestroyPipelineLayout(device, PipelineLayout, nullptr); PipelineLayout = VK_NULL_HANDLE;}
}

void graphics::OutliningExtension::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass){
    auto vertShaderCode = ShaderModule::readFile(ExternalPath + "core\\deferredGraphics\\shaders\\outlining\\outliningVert.spv");
    auto fragShaderCode = ShaderModule::readFile(ExternalPath + "core\\deferredGraphics\\shaders\\outlining\\outliningFrag.spv");
    VkShaderModule vertShaderModule = ShaderModule::create(&device, vertShaderCode);
    VkShaderModule fragShaderModule = ShaderModule::create(&device, fragShaderCode);
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        vkDefault::vertrxShaderStage(vertShaderModule),
        vkDefault::fragmentShaderStage(fragShaderModule)
    };

    auto bindingDescription = gltfModel::Vertex::getBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkViewport viewport = vkDefault::viewport(pInfo->Extent);
    VkRect2D scissor = vkDefault::scissor(pInfo->Extent);
    VkPipelineViewportStateCreateInfo viewportState = vkDefault::viewportState(&viewport, &scissor);
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = vkDefault::rasterizationState(VK_FRONT_FACE_COUNTER_CLOCKWISE);
    VkPipelineMultisampleStateCreateInfo multisampling = vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = vkDefault::depthStencilDisable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {
        vkDefault::colorBlendAttachmentState(VK_FALSE),
        vkDefault::colorBlendAttachmentState(VK_FALSE),
        vkDefault::colorBlendAttachmentState(VK_FALSE),
        vkDefault::colorBlendAttachmentState(VK_FALSE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(OutliningPushConst);
    std::vector<VkDescriptorSetLayout> SetLayouts = {
        Parent->SceneDescriptorSetLayout,
        Parent->ObjectDescriptorSetLayout,
        Parent->PrimitiveDescriptorSetLayout,
        Parent->MaterialDescriptorSetLayout
    };
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &PipelineLayout);

    std::vector<VkGraphicsPipelineCreateInfo> pipelineInfo;
    pipelineInfo.push_back(VkGraphicsPipelineCreateInfo{});
        pipelineInfo.back().sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.back().pNext = nullptr;
        pipelineInfo.back().stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.back().pStages = shaderStages.data();
        pipelineInfo.back().pVertexInputState = &vertexInputInfo;
        pipelineInfo.back().pInputAssemblyState = &inputAssembly;
        pipelineInfo.back().pViewportState = &viewportState;
        pipelineInfo.back().pRasterizationState = &rasterizer;
        pipelineInfo.back().pMultisampleState = &multisampling;
        pipelineInfo.back().pColorBlendState = &colorBlending;
        pipelineInfo.back().layout = PipelineLayout;
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline);

    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
}

void graphics::OutliningExtension::render(uint32_t frameNumber, VkCommandBuffer commandBuffers)
{
    vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    for(auto object: Parent->objects)
    {
        if(VkDeviceSize offsets = 0; object->getEnable()&&object->getOutliningEnable()){
            vkCmdBindVertexBuffers(commandBuffers, 0, 1, & object->getModel(frameNumber)->vertices.buffer, &offsets);
            if (object->getModel(frameNumber)->indices.buffer != VK_NULL_HANDLE){
                vkCmdBindIndexBuffer(commandBuffers, object->getModel(frameNumber)->indices.buffer, 0, VK_INDEX_TYPE_UINT32);
            }

            OutliningPushConst pushConst{};
                pushConst.stencilColor = object->getOutliningColor();
                pushConst.width = object->getOutliningWidth();
            vkCmdPushConstants(commandBuffers, PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(OutliningPushConst), &pushConst);

            uint32_t primiriveCount = 0;
            for (auto node : object->getModel(frameNumber)->nodes){
                std::vector<VkDescriptorSet> descriptorSets = {Parent->DescriptorSets[frameNumber],object->getDescriptorSet()[frameNumber]};
                renderNode(commandBuffers,node,PipelineLayout,static_cast<uint32_t>(descriptorSets.size()),descriptorSets.data(),primiriveCount);
            }
        }
    }
}

void graphics::OutliningExtension::renderNode(VkCommandBuffer commandBuffer, Node *node, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount)
{
    if (node->mesh)
    {
        for (Primitive* primitive : node->mesh->primitives)
        {
            std::vector<VkDescriptorSet> nodeDescriptorSets(descriptorSetsCount);
            std::copy(descriptorSets, descriptorSets + descriptorSetsCount, nodeDescriptorSets.data());
            nodeDescriptorSets.push_back(node->mesh->uniformBuffer.descriptorSet);
            nodeDescriptorSets.push_back(primitive->material.descriptorSet);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSetsCount+2, nodeDescriptorSets.data(), 0, NULL);

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
