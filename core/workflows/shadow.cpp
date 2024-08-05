#include "shadow.h"
#include "operations.h"
#include "vkdefault.h"
#include "light.h"
#include "model.h"
#include "depthMap.h"

namespace moon::workflows {

ShadowGraphics::ShadowGraphics(ShadowGraphicsParameters& parameters, const interfaces::Objects* objects, interfaces::DepthMaps* depthMaps) :
    parameters(parameters), shadow(parameters, objects, depthMaps)
{}

void ShadowGraphics::createRenderPass()
{
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = { utils::Attachments::depthDescription(VK_FORMAT_D32_SFLOAT)};

    utils::SubpassInfos subpassInfos;
    auto& subpass = subpassInfos.emplace_back();
    subpass.depth = { VkAttachmentReference{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL} };

    renderPass = utils::vkDefault::RenderPass(device, attachments, subpassInfos, {});
}

void ShadowGraphics::Shadow::create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) {
    lightUniformBufferSetLayout = interfaces::Light::createBufferDescriptorSetLayout(device);
    objectDescriptorSetLayout = interfaces::Object::createBaseDescriptorSetLayout(device);
    primitiveDescriptorSetLayout = interfaces::Model::createNodeDescriptorSetLayout(device);
    materialDescriptorSetLayout = interfaces::Model::createMaterialDescriptorSetLayout(device);

    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, parameters.shadersPath / shadersNames.at(workflows::ShaderType::Vertex));
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader };

    auto bindingDescription = interfaces::Model::Vertex::getBindingDescription();
    auto attributeDescriptions = interfaces::Model::Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkViewport viewport = utils::vkDefault::viewport({0,0}, parameters.imageInfo.Extent);
    VkRect2D scissor = utils::vkDefault::scissor({0,0}, parameters.imageInfo.Extent);
    VkPipelineViewportStateCreateInfo viewportState = utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = utils::vkDefault::rasterizationState();
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 4.0f;
    rasterizer.depthBiasSlopeFactor = 1.5f;
    VkPipelineMultisampleStateCreateInfo multisampling = utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = utils::vkDefault::depthStencilEnable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {utils::vkDefault::colorBlendAttachmentState(VK_FALSE)};
    VkPipelineColorBlendStateCreateInfo colorBlending = utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(interfaces::MaterialBlock);
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
        lightUniformBufferSetLayout,
        objectDescriptorSetLayout,
        primitiveDescriptorSetLayout,
        materialDescriptorSetLayout
    };
    pipelineLayout = utils::vkDefault::PipelineLayout(device, descriptorSetLayouts, pushConstantRange);

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
        pipelineInfo.back().renderPass = renderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);
}

void ShadowGraphics::create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase&) {
    commandBuffers = commandPool.allocateCommandBuffers(parameters.imageInfo.Count);
    if(parameters.enable && !created){
        createRenderPass();
        const workflows::ShaderNames shaderNames = {
            {workflows::ShaderType::Vertex, "shadow/shadowMapVert.spv"}
        };
        shadow.create(shaderNames, device, renderPass);
        created = true;
    }
    for (auto& [light, depthMap] : *shadow.depthMaps) {
        depthMap.update(light->isShadowEnable() && parameters.enable);
    }
}

void ShadowGraphics::updateFramebuffersMap() {
    FramebuffersMap newFramebuffersMap;
    for (const auto& [light, depthMap] : *shadow.depthMaps) {
        if (framebuffersMap.find(&depthMap) != framebuffersMap.end()) {
            newFramebuffersMap[&depthMap] = std::move(framebuffersMap[&depthMap]);
        }
    }
    std::swap(framebuffersMap, newFramebuffersMap);
}

void ShadowGraphics::updateCommandBuffer(uint32_t frameNumber) {
    if (!parameters.enable || !created) return;

    updateFramebuffersMap();

    for(const auto& [light, depthMap] : *shadow.depthMaps){
        if (!light->isShadowEnable()){
            continue;
        }

        if(framebuffersMap.find(&depthMap) == framebuffersMap.end()) {
            framebuffersMap[&depthMap].resize(parameters.imageInfo.Count);
            for (size_t i = 0; i < parameters.imageInfo.Count; i++) {
                VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &depthMap.attachments().imageView(i);
                framebufferInfo.width = parameters.imageInfo.Extent.width;
                framebufferInfo.height = parameters.imageInfo.Extent.height;
                framebufferInfo.layers = 1;
                framebuffersMap[&depthMap][i] = utils::vkDefault::Framebuffer(device, framebufferInfo);
            }
        }

        render(frameNumber, commandBuffers[frameNumber], light, depthMap, framebuffersMap[&depthMap][frameNumber]);
    }
}

void ShadowGraphics::render(uint32_t frameNumber, VkCommandBuffer commandBuffer, interfaces::Light* lightSource, const utils::DepthMap& depthMap, const utils::vkDefault::Framebuffer& framebuffer)
{
    std::vector<VkClearValue> clearValues;
    clearValues.push_back(depthMap.attachments().clearValue());

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffer;
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = parameters.imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.pipeline);
    for(const auto& object: *shadow.objects){
        if(VkDeviceSize offsets = 0; (interfaces::ObjectType::base & object->pipelineFlagBits()) && object->getEnable() && object->getEnableShadow()){
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, object->model()->getVertices(), &offsets);
            if (object->model()->getIndices() != VK_NULL_HANDLE){
                vkCmdBindIndexBuffer(commandBuffer, *object->model()->getIndices(), 0, VK_INDEX_TYPE_UINT32);
            }

            utils::vkDefault::DescriptorSets descriptorSets = {lightSource->getDescriptorSet(frameNumber), object->getDescriptorSet(frameNumber)};

            interfaces::MaterialBlock material{};

            uint32_t primitives = 0;
            object->model()->render(
                        object->getInstanceNumber(frameNumber),
                        commandBuffer,
                        shadow.pipelineLayout,
                        static_cast<uint32_t>(descriptorSets.size()),
                        descriptorSets.data(),primitives,
                        sizeof(interfaces::MaterialBlock),
                        0,
                        &material);
        }
    }

    vkCmdEndRenderPass(commandBuffer);
}

}
