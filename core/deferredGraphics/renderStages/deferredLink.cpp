#include "deferredLink.h"
#include "operations.h"
#include "vkdefault.h"

namespace moon::deferredGraphics {

void Link::setDeviceProp(VkDevice device){
    this->device = device;
}

void Link::setImageCount(const uint32_t& count){
    this->imageCount = count;
}

void Link::setShadersPath(const std::filesystem::path &shadersPath){
    this->shadersPath = shadersPath;
}

void Link::setRenderPass(VkRenderPass renderPass) {
    this->renderPass = renderPass;
}

void Link::setPositionInWindow(const moon::math::Vector<float,2>& offset, const moon::math::Vector<float,2>& size){
    pushConstant.offset = offset;
    pushConstant.size = size;
}

void Link::createDescriptorSetLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);
}

void Link::createPipeline(moon::utils::ImageInfo* pInfo) {
    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, shadersPath / "linkable/deferredLinkableVert.spv");
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, shadersPath / "linkable/deferredLinkableFrag.spv");
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader, fragShader };

    VkViewport viewport = moon::utils::vkDefault::viewport({0,0}, pInfo->Extent);
    VkRect2D scissor = moon::utils::vkDefault::scissor({0,0}, pInfo->Extent);
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
        pushConstantRange.back().size = sizeof(PushConstant);
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { descriptorSetLayout };
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

void Link::createDescriptorPool() {
    descriptorPool = utils::vkDefault::DescriptorPool(device, {&descriptorSetLayout}, imageCount);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageCount);
}

void Link::updateDescriptorSets(const moon::utils::Attachments* attachment) {
    for (size_t image = 0; image < this->imageCount; image++)
    {
        VkDescriptorImageInfo imageInfo;
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = attachment->imageView(image);
            imageInfo.sampler = attachment->sampler();

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptorSets[image];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &imageInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void Link::draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const
{
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant), &pushConstant);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[imageNumber], 0, nullptr);
    vkCmdDraw(commandBuffer, 6, 1, 0, 0);
}

}
