#include "shadow.h"
#include "operations.h"
#include "vkdefault.h"
#include "light.h"
#include "object.h"
#include "model.h"
#include "depthMap.h"

namespace moon::workflows {

moon::utils::Attachments* ShadowGraphics::createAttachments()
{
    moon::utils::Attachments* pAttachments = new moon::utils::Attachments;
    pAttachments->createDepth(physicalDevice,device,image.Format,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,image.Extent,image.Count);
    VkSamplerCreateInfo samplerInfo = moon::utils::vkDefault::samler();
    CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &pAttachments->sampler));
    return pAttachments;
}

void ShadowGraphics::Shadow::destroy(VkDevice device)
{
    Workbody::destroy(device);
}

ShadowGraphics::ShadowGraphics(bool enable, std::vector<moon::interfaces::Object*>* objects, std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap*>* depthMaps) :
    enable(enable)
{
    shadow.objects = objects;
    shadow.depthMaps = depthMaps;
}

void ShadowGraphics::destroy()
{
    shadow.destroy(device);
    Workflow::destroy();
}

void ShadowGraphics::createRenderPass()
{
    VkAttachmentDescription attachments = moon::utils::Attachments::depthDescription(VK_FORMAT_D32_SFLOAT);
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
    CHECK(vkCreateRenderPass(device, &renderPassInfo, NULL, &renderPass));
}

void ShadowGraphics::createPipelines()
{
    shadow.vertShaderPath = shadersPath / "shadow/shadowMapVert.spv";
    shadow.createDescriptorSetLayout(device);
    shadow.createPipeline(device,&image,renderPass);
}

void ShadowGraphics::Shadow::createDescriptorSetLayout(VkDevice device)
{
    lightUniformBufferSetLayout = moon::interfaces::Light::createBufferDescriptorSetLayout(device);
    objectDescriptorSetLayout = moon::interfaces::Object::createDescriptorSetLayout(device);
    primitiveDescriptorSetLayout = moon::interfaces::Model::createNodeDescriptorSetLayout(device);
    materialDescriptorSetLayout = moon::interfaces::Model::createMaterialDescriptorSetLayout(device);
}

void ShadowGraphics::Shadow::createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass) {
    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, vertShaderPath);
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader };

    auto bindingDescription = moon::interfaces::Model::Vertex::getBindingDescription();
    auto attributeDescriptions = moon::interfaces::Model::Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkViewport viewport = moon::utils::vkDefault::viewport({0,0}, pInfo->Extent);
    VkRect2D scissor = moon::utils::vkDefault::scissor({0,0}, pInfo->Extent);
    VkPipelineViewportStateCreateInfo viewportState = moon::utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = moon::utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = moon::utils::vkDefault::rasterizationState();
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 4.0f;
    rasterizer.depthBiasSlopeFactor = 1.5f;
    VkPipelineMultisampleStateCreateInfo multisampling = moon::utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = moon::utils::vkDefault::depthStencilEnable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {moon::utils::vkDefault::colorBlendAttachmentState(VK_FALSE)};
    VkPipelineColorBlendStateCreateInfo colorBlending = moon::utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(moon::interfaces::MaterialBlock);
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
        lightUniformBufferSetLayout,
        objectDescriptorSetLayout,
        primitiveDescriptorSetLayout,
        materialDescriptorSetLayout
    };
    CHECK(pipelineLayout.create(device, descriptorSetLayouts, pushConstantRange));

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
        pipelineInfo.back().layout = pipelineLayout;
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    CHECK(pipeline.create(device, pipelineInfo));
}

void ShadowGraphics::createFramebuffers(moon::utils::DepthMap* depthMap)
{
    depthMap->get() = createAttachments();
    depthMap->updateDescriptorSets(device, image.Count);
    framebuffers[depthMap].resize(image.Count);
    for (size_t j = 0; j < image.Count; j++){
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &depthMap->get()->instances[j].imageView;
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[depthMap][j]));
    }
}


void ShadowGraphics::destroyFramebuffers(moon::utils::DepthMap* depthMap)
{
    if(depthMap->get()){
        depthMap->get()->deleteAttachment(device);
        depthMap->get()->deleteSampler(device);
    }
    if(framebuffers.count(depthMap)){
        for(auto& frame: framebuffers[depthMap]){
            if(frame){ vkDestroyFramebuffer(device, frame,nullptr); frame = VK_NULL_HANDLE;}
        }
        framebuffers.erase(depthMap);
    }
}

void ShadowGraphics::create(moon::utils::AttachmentsDatabase&)
{
    if(enable){
        createRenderPass();
        createPipelines();
    }
}

void ShadowGraphics::updateCommandBuffer(uint32_t frameNumber)
{
    if(!enable) return;

    for(const auto& [light, depth] : *shadow.depthMaps){
        render(frameNumber, commandBuffers[frameNumber], light, depth);
    }
}

void ShadowGraphics::render(uint32_t frameNumber, VkCommandBuffer commandBuffer, moon::interfaces::Light* lightSource, moon::utils::DepthMap* depthMap)
{
    std::vector<VkClearValue> clearValues;
    clearValues.push_back(VkClearValue{depthMap->get()->clearValue.color});

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[depthMap][frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.pipeline);
    for(auto object: *shadow.objects){
        if(VkDeviceSize offsets = 0; (moon::interfaces::ObjectType::base & object->getPipelineBitMask()) && object->getEnable() && object->getEnableShadow()){
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, object->getModel()->getVertices(), &offsets);
            if (object->getModel()->getIndices() != VK_NULL_HANDLE){
                vkCmdBindIndexBuffer(commandBuffer, *object->getModel()->getIndices(), 0, VK_INDEX_TYPE_UINT32);
            }

            std::vector<VkDescriptorSet> descriptorSets = {
                lightSource->getDescriptorSets()[frameNumber],
                object->getDescriptorSet()[frameNumber]
            };

            moon::interfaces::MaterialBlock material{};

            uint32_t primitives = 0;
            object->getModel()->render(
                        object->getInstanceNumber(frameNumber),
                        commandBuffer,
                        shadow.pipelineLayout,
                        static_cast<uint32_t>(descriptorSets.size()),
                        descriptorSets.data(),primitives,
                        sizeof(moon::interfaces::MaterialBlock),
                        0,
                        &material);
        }
    }

    vkCmdEndRenderPass(commandBuffer);
}

}
