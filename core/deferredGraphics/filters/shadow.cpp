#include "shadow.h"
#include "../../utils/operations.h"
#include "../utils/vkdefault.h"
#include "../../transformational/lightInterface.h"
#include "../../transformational/object.h"
#include "../../models/gltfmodel.h"

void shadowGraphics::createAttachments(uint32_t attachmentsCount, attachments* pAttachments)
{
    static_cast<void>(attachmentsCount);
    pAttachments->createDepth(&physicalDevice,&device,image.Format,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,image.Extent,image.Count);
    VkSamplerCreateInfo SamplerInfo{};
        SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    vkCreateSampler(device, &SamplerInfo, nullptr, &pAttachments->sampler);
}

void shadowGraphics::Shadow::destroy(VkDevice device)
{
    filter::destroy(device);
    if(lightUniformBufferSetLayout) {vkDestroyDescriptorSetLayout(device, lightUniformBufferSetLayout, nullptr); lightUniformBufferSetLayout = VK_NULL_HANDLE;}
    if(uniformBlockSetLayout)       {vkDestroyDescriptorSetLayout(device, uniformBlockSetLayout, nullptr); uniformBlockSetLayout = VK_NULL_HANDLE;}
    if(uniformBufferSetLayout)      {vkDestroyDescriptorSetLayout(device, uniformBufferSetLayout, nullptr); uniformBufferSetLayout = VK_NULL_HANDLE;}
}

void shadowGraphics::destroy()
{
    shadow.destroy(device);

    filterGraphics::destroy();
}

void shadowGraphics::createRenderPass()
{
    VkAttachmentDescription attachments = attachments::depthDescription(VK_FORMAT_D32_SFLOAT);
    VkAttachmentReference depthRef{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.pDepthStencilAttachment = &depthRef;

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &attachments;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
    vkCreateRenderPass(device, &renderPassInfo, NULL, &renderPass);
}

void shadowGraphics::createPipelines()
{
    shadow.vertShaderPath = externalPath + "core\\deferredGraphics\\shaders\\shadow\\shad.spv";
    shadow.createDescriptorSetLayout(device);
    shadow.createPipeline(device,&image,renderPass);
}

void shadowGraphics::Shadow::createDescriptorSetLayout(VkDevice device)
{
    SpotLight::createShadowDescriptorSetLayout(device, &lightUniformBufferSetLayout);
    object::createDescriptorSetLayout(device, &uniformBufferSetLayout);
    gltfModel::createNodeDescriptorSetLayout(device, &uniformBlockSetLayout);
}

void shadowGraphics::Shadow::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass)
{
    auto vertShaderCode = ShaderModule::readFile(vertShaderPath);
    VkShaderModule vertShaderModule = ShaderModule::create(&device,vertShaderCode);
    VkPipelineShaderStageCreateInfo vertShaderStageInfo = vkDefault::vertrxShaderStage(vertShaderModule);
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {vertShaderStageInfo};

    auto bindingDescription = gltfModel::Vertex::getShadowBindingDescription();
    auto attributeDescriptions = gltfModel::Vertex::getShadowAttributeDescriptions();
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
    VkPipelineRasterizationStateCreateInfo rasterizer = vkDefault::rasterizationState();
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 4.0f;
    rasterizer.depthBiasSlopeFactor = 1.5f;
    VkPipelineMultisampleStateCreateInfo multisampling = vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = vkDefault::depthStencilEnable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {vkDefault::colorBlendAttachmentState(VK_FALSE)};
    VkPipelineColorBlendStateCreateInfo colorBlending = vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::array<VkDescriptorSetLayout,3> SetLayouts = {lightUniformBufferSetLayout,uniformBufferSetLayout,uniformBlockSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &PipelineLayout);

    VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext = nullptr;
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = PipelineLayout;
        pipelineInfo.renderPass = pRenderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.pDepthStencilState = &depthStencil;
    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Pipeline);

    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void shadowGraphics::bindBaseObject(object *newObject)
{
    shadow.objects.push_back(newObject);
}

bool shadowGraphics::removeBaseObject(object* object)
{
    bool result = false;
    for(uint32_t index = 0; index<shadow.objects.size(); index++){
        if(object==shadow.objects[index]){
            shadow.objects.erase(shadow.objects.begin()+index);
            result = true;
        }
    }
    return result;
}

void shadowGraphics::createFramebuffers(light* lightSource)
{
    framebuffers[lightSource].resize(image.Count);
    for (size_t j = 0; j < image.Count; j++){
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &lightSource->getAttachments()->imageView[j];
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[lightSource][j]);
    }
}

void shadowGraphics::addLightSource(light* lightSource)
{
    shadow.lightSources.push_back(lightSource);
}

void shadowGraphics::removeLightSource(light* lightSource)
{
    for(uint32_t index = 0; index<shadow.lightSources.size(); index++){
        if(lightSource==shadow.lightSources[index]){

            for(uint32_t i=0;i<image.Count;i++){
                if(framebuffers[shadow.lightSources[index]][i]){
                    vkDestroyFramebuffer(device, framebuffers[shadow.lightSources[index]][i],nullptr);
                    framebuffers[shadow.lightSources[index]][i] = VK_NULL_HANDLE;
                }
            }
            framebuffers.erase(shadow.lightSources[index]);
            shadow.lightSources.erase(shadow.lightSources.begin()+index);
            break;
        }
    }
}

void shadowGraphics::updateCommandBuffer(uint32_t frameNumber)
{
    for(uint32_t attachmentNumber = 0; attachmentNumber < shadow.lightSources.size(); attachmentNumber++){
        render(frameNumber,commandBuffers[frameNumber],attachmentNumber);
    }
}

void shadowGraphics::render(uint32_t frameNumber, VkCommandBuffer commandBuffer, uint32_t attachmentNumber)
{
    std::vector<VkClearValue> clearValues;
    clearValues.push_back(VkClearValue{shadow.lightSources[attachmentNumber]->getAttachments()->clearValue.color});

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[shadow.lightSources[attachmentNumber]][frameNumber];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.Pipeline);
        for(auto object: shadow.objects)
        {
            if(object->getEnable()&&object->getEnableShadow()){
                VkDeviceSize offsets[1] = { 0 };
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, & object->getModel(frameNumber)->vertices.buffer, offsets);
                if (object->getModel(frameNumber)->indices.buffer != VK_NULL_HANDLE)
                    vkCmdBindIndexBuffer(commandBuffer,  object->getModel(frameNumber)->indices.buffer, 0, VK_INDEX_TYPE_UINT32);

                for (auto node : object->getModel(frameNumber)->nodes){
                    std::vector<VkDescriptorSet> descriptorSets = {shadow.lightSources[attachmentNumber]->getShadowDescriptorSets()[frameNumber],object->getDescriptorSet()[frameNumber]};
                    renderNode(commandBuffer,node,static_cast<uint32_t>(descriptorSets.size()),descriptorSets.data());
                }
            }
        }

    vkCmdEndRenderPass(commandBuffer);
}

void shadowGraphics::renderNode(VkCommandBuffer commandBuffer, Node* node, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets)
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
