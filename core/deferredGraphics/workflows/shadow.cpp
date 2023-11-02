#include "shadow.h"
#include "operations.h"
#include "vkdefault.h"
#include "light.h"
#include "object.h"
#include "model.h"

void shadowGraphics::createAttachments(uint32_t attachmentsCount, attachments* pAttachments)
{
    static_cast<void>(attachmentsCount);
    pAttachments->createDepth(physicalDevice,device,image.Format,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,image.frameBufferExtent,image.Count);
    VkSamplerCreateInfo samplerInfo = vkDefault::samler();
    vkCreateSampler(device, &samplerInfo, nullptr, &pAttachments->sampler);
}

void shadowGraphics::Shadow::destroy(VkDevice device)
{
    workbody::destroy(device);
    if(lightUniformBufferSetLayout)     {vkDestroyDescriptorSetLayout(device, lightUniformBufferSetLayout, nullptr); lightUniformBufferSetLayout = VK_NULL_HANDLE;}
    if(ObjectDescriptorSetLayout)       {vkDestroyDescriptorSetLayout(device, ObjectDescriptorSetLayout, nullptr); ObjectDescriptorSetLayout = VK_NULL_HANDLE;}
    if(PrimitiveDescriptorSetLayout)    {vkDestroyDescriptorSetLayout(device, PrimitiveDescriptorSetLayout, nullptr); PrimitiveDescriptorSetLayout = VK_NULL_HANDLE;}
    if(MaterialDescriptorSetLayout)     {vkDestroyDescriptorSetLayout(device, MaterialDescriptorSetLayout, nullptr); MaterialDescriptorSetLayout = VK_NULL_HANDLE;}
}

shadowGraphics::shadowGraphics(bool enable) :
    enable(enable)
{}

void shadowGraphics::destroy()
{
    shadow.destroy(device);
    workflow::destroy();
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
    shadow.vertShaderPath = shadersPath / "shadow/shad.spv";
    shadow.createDescriptorSetLayout(device);
    shadow.createPipeline(device,&image,renderPass);
}

void shadowGraphics::Shadow::createDescriptorSetLayout(VkDevice device)
{
    light::createBufferDescriptorSetLayout(device, &lightUniformBufferSetLayout);
    object::createDescriptorSetLayout(device, &ObjectDescriptorSetLayout);
    model::createNodeDescriptorSetLayout(device, &PrimitiveDescriptorSetLayout);
    model::createMaterialDescriptorSetLayout(device, &MaterialDescriptorSetLayout);
}

void shadowGraphics::Shadow::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass)
{
    auto vertShaderCode = ShaderModule::readFile(vertShaderPath);
    VkShaderModule vertShaderModule = ShaderModule::create(&device,vertShaderCode);
    VkPipelineShaderStageCreateInfo vertShaderStageInfo = vkDefault::vertrxShaderStage(vertShaderModule);
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {vertShaderStageInfo};

    auto bindingDescription = model::Vertex::getBindingDescription();
    auto attributeDescriptions = model::Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkViewport viewport = vkDefault::viewport(pInfo->Offset, pInfo->Extent);
    VkRect2D scissor = vkDefault::scissor({0,0}, pInfo->frameBufferExtent);
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

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(MaterialBlock);
    std::vector<VkDescriptorSetLayout> SetLayouts = {
        lightUniformBufferSetLayout,
        ObjectDescriptorSetLayout,
        PrimitiveDescriptorSetLayout,
        MaterialDescriptorSetLayout
    };
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
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
            framebufferInfo.pAttachments = &lightSource->getShadowMaps().back()->instances[j].imageView;
            framebufferInfo.width = image.frameBufferExtent.width;
            framebufferInfo.height = image.frameBufferExtent.height;
            framebufferInfo.layers = 1;
        vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[lightSource][j]);
    }
}

void shadowGraphics::bindLightSource(light* lightSource)
{
    shadow.lightSources.push_back(lightSource);
    if(lightSource->getShadowMaps().empty()){
        lightSource->getShadowMaps().push_back(new attachments());
        createAttachments(1, lightSource->getShadowMaps().back());
    }
}

void shadowGraphics::removeLightSource(light* lightSource)
{
    for(uint32_t index = 0; index<shadow.lightSources.size(); index++){
        if(lightSource==shadow.lightSources[index]){
            for(auto shadowMap : lightSource->getShadowMaps()){
                if(shadowMap){
                    shadowMap->deleteAttachment(device);
                    shadowMap->deleteSampler(device);
                    delete shadowMap;
                }
            }
            lightSource->getShadowMaps().clear();

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

void shadowGraphics::create(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>&)
{
    if(enable){
        createRenderPass();
        createPipelines();
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
    clearValues.push_back(VkClearValue{shadow.lightSources[attachmentNumber]->getShadowMaps().back()->clearValue.color});

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[shadow.lightSources[attachmentNumber]][frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = image.frameBufferExtent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.Pipeline);
        for(auto object: shadow.objects){
            if(VkDeviceSize offsets = 0; object->getEnable() && object->getEnableShadow()){
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, object->getModel()->getVertices(), &offsets);
                if (object->getModel()->getIndices() != VK_NULL_HANDLE){
                    vkCmdBindIndexBuffer(commandBuffer, *object->getModel()->getIndices(), 0, VK_INDEX_TYPE_UINT32);
                }

                std::vector<VkDescriptorSet> descriptorSets = {
                    shadow.lightSources[attachmentNumber]->getDescriptorSets()[frameNumber],
                    object->getDescriptorSet()[frameNumber]
                };

                MaterialBlock material{};

                uint32_t primitives = 0;
                object->getModel()->render(
                            object->getInstanceNumber(frameNumber),
                            commandBuffer,
                            shadow.PipelineLayout,
                            static_cast<uint32_t>(descriptorSets.size()),
                            descriptorSets.data(),primitives,
                            sizeof(MaterialBlock),
                            0,
                            &material);
            }
        }

    vkCmdEndRenderPass(commandBuffer);
}
