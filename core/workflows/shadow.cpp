#include "shadow.h"
#include "operations.h"
#include "vkdefault.h"
#include "light.h"
#include "object.h"
#include "model.h"
#include "depthMap.h"

namespace moon::workflows {

ShadowGraphics::ShadowGraphics(const moon::utils::ImageInfo& imageInfo, const std::filesystem::path& shadersPath, bool enable, std::vector<moon::interfaces::Object*>* objects, std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap>* depthMaps)
    : Workflow(imageInfo, shadersPath), enable(enable), shadow(this->imageInfo)
{
    shadow.objects = objects;
    shadow.depthMaps = depthMaps;
}

void ShadowGraphics::createRenderPass()
{
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = { moon::utils::Attachments::depthDescription(VK_FORMAT_D32_SFLOAT)};
    VkAttachmentReference depthRef{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    utils::vkDefault::RenderPass::SubpassDescriptions subpasses;
    subpasses.push_back(VkSubpassDescription{});
    subpasses.back().pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses.back().pDepthStencilAttachment = &depthRef;

    renderPass = utils::vkDefault::RenderPass(device, attachments, subpasses, {});
}

void ShadowGraphics::Shadow::create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) {
    this->vertShaderPath = vertShaderPath;
    this->fragShaderPath = fragShaderPath;
    this->device = device;

    lightUniformBufferSetLayout = moon::interfaces::Light::createBufferDescriptorSetLayout(device);
    objectDescriptorSetLayout = moon::interfaces::Object::createDescriptorSetLayout(device);
    primitiveDescriptorSetLayout = moon::interfaces::Model::createNodeDescriptorSetLayout(device);
    materialDescriptorSetLayout = moon::interfaces::Model::createMaterialDescriptorSetLayout(device);

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

    VkViewport viewport = moon::utils::vkDefault::viewport({0,0}, imageInfo.Extent);
    VkRect2D scissor = moon::utils::vkDefault::scissor({0,0}, imageInfo.Extent);
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
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);
}

void ShadowGraphics::create(moon::utils::AttachmentsDatabase&)
{
    if(enable){
        createRenderPass();
        shadow.create(shadersPath / "shadow/shadowMapVert.spv", "", device, renderPass);
    }
    for (auto& [light, depthMap] : *shadow.depthMaps) {
        depthMap.update(light->isShadowEnable() && enable);
    }
}

void ShadowGraphics::updateCommandBuffer(uint32_t frameNumber)
{
    if(!enable) return;

    for(const auto& [light, depthMap] : *shadow.depthMaps){
        if (light->isShadowEnable() && framebuffersMap.find(&depthMap) == framebuffersMap.end()){
            framebuffersMap[&depthMap].resize(imageInfo.Count);
            for (size_t i = 0; i < imageInfo.Count; i++) {
                VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &depthMap.attachments().imageView(i);
                framebufferInfo.width = imageInfo.Extent.width;
                framebufferInfo.height = imageInfo.Extent.height;
                framebufferInfo.layers = 1;
                framebuffersMap[&depthMap][i] = utils::vkDefault::Framebuffer(device, framebufferInfo);
            }
        }
        render(frameNumber, commandBuffers[frameNumber], light, depthMap);
    }
}

void ShadowGraphics::render(uint32_t frameNumber, VkCommandBuffer commandBuffer, moon::interfaces::Light* lightSource, const moon::utils::DepthMap& depthMap)
{
    std::vector<VkClearValue> clearValues;
    clearValues.push_back(depthMap.attachments().clearValue());

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffersMap[&depthMap][frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.pipeline);
    for(const auto& object: *shadow.objects){
        if(VkDeviceSize offsets = 0; (moon::interfaces::ObjectType::base & object->getPipelineBitMask()) && object->getEnable() && object->getEnableShadow()){
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, object->getModel()->getVertices(), &offsets);
            if (object->getModel()->getIndices() != VK_NULL_HANDLE){
                vkCmdBindIndexBuffer(commandBuffer, *object->getModel()->getIndices(), 0, VK_INDEX_TYPE_UINT32);
            }

            utils::vkDefault::DescriptorSets descriptorSets = {lightSource->getDescriptorSets()[frameNumber], object->getDescriptorSet(frameNumber)};

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
