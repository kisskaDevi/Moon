#include "rayTracingLink.h"
#include "operations.h"
#include "vkdefault.h"

namespace moon::rayTracingGraphics {

void RayTracingLink::setRenderPass(VkRenderPass renderPass)
{
    this->renderPass = renderPass;
}

void RayTracingLink::setPositionInWindow(const moon::math::Vector<float,2>& offset, const moon::math::Vector<float,2>& size){
    pushConstant.offset = offset;
    pushConstant.size = size;
}

void RayTracingLink::createDescriptorSetLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    CHECK(descriptorSetLayout.create(device, bindings));
}

void RayTracingLink::createPipeline() {
    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, shadersPath / "linkable/linkableVert.spv");
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, shadersPath / "linkable/linkableFrag.spv");
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader, fragShader };

    VkViewport viewport = moon::utils::vkDefault::viewport({0,0}, imageInfo.Extent);
    VkRect2D scissor = moon::utils::vkDefault::scissor({0,0}, imageInfo.Extent);
    VkPipelineViewportStateCreateInfo viewportState = moon::utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = moon::utils::vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = moon::utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = moon::utils::vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = moon::utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = moon::utils::vkDefault::depthStencilDisable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {moon::utils::vkDefault::colorBlendAttachmentState(VK_FALSE)};
    VkPipelineColorBlendStateCreateInfo colorBlending = moon::utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
    pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
    pushConstantRange.back().offset = 0;
    pushConstantRange.back().size = sizeof(LinkPushConstant);
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { descriptorSetLayout };
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
    pipelineInfo.back().renderPass = renderPass;
    pipelineInfo.back().subpass = 0;
    pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.back().pDepthStencilState = &depthStencil;
    CHECK(pipeline.create(device, pipelineInfo));
}

void RayTracingLink::createDescriptors() {
    CHECK(descriptorPool.create(device, { &descriptorSetLayout }, imageInfo.Count));
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageInfo.Count);
}

void RayTracingLink::updateDescriptorSets(const moon::utils::AttachmentsDatabase& aDatabase) {
    for (size_t image = 0; image < this->imageInfo.Count; image++)
    {
        VkDescriptorImageInfo imageInfo = aDatabase.descriptorImageInfo(parameters.in.color, image);
        VkDescriptorImageInfo bloomImageInfo = aDatabase.descriptorImageInfo(parameters.in.bloom, image);
        VkDescriptorImageInfo bbImageInfo = aDatabase.descriptorImageInfo(parameters.in.boundingBox, image);

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptorSets[image];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &imageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptorSets[image];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &bbImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptorSets[image];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &bloomImageInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}


void RayTracingLink::create(const std::filesystem::path& shadersPath, VkDevice device, const moon::utils::ImageInfo& imageInfo) {
    this->device = device;
    this->imageInfo = imageInfo;
    this->shadersPath = shadersPath;

    createDescriptorSetLayout();
    createPipeline();
    createDescriptors();
}

void RayTracingLink::draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const
{
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(LinkPushConstant), &pushConstant);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[imageNumber], 0, nullptr);
    vkCmdDraw(commandBuffer, 6, 1, 0, 0);
}

}
