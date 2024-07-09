#include "bloom.h"
#include "operations.h"
#include "vkdefault.h"

namespace moon::workflows {

struct BloomPushConst{
    alignas (4) float deltax;
    alignas (4) float deltay;
    alignas (4) float blitFactor;
};

BloomGraphics::BloomGraphics(const moon::utils::ImageInfo& imageInfo, const std::filesystem::path& shadersPath, BloomParameters& parameters)
    : Workflow(imageInfo, shadersPath), parameters(parameters), filter(this->imageInfo), bloom(this->imageInfo, parameters)
{}

void BloomGraphics::createRenderPass(){
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        moon::utils::Attachments::imageDescription(imageInfo.Format,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    };

    std::vector<std::vector<VkAttachmentReference>> attachmentRef;
    attachmentRef.push_back(std::vector<VkAttachmentReference>());
        attachmentRef.back().push_back(VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    utils::vkDefault::RenderPass::SubpassDescriptions subpasses;
    for(auto refIt = attachmentRef.begin(); refIt != attachmentRef.end(); refIt++){
        subpasses.push_back(VkSubpassDescription{});
        subpasses.back().pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpasses.back().colorAttachmentCount = static_cast<uint32_t>(refIt->size());
        subpasses.back().pColorAttachments = refIt->data();
    }

    utils::vkDefault::RenderPass::SubpassDependencies dependencies;
    dependencies.push_back(VkSubpassDependency{});
        dependencies.back().srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies.back().dstSubpass = 0;
        dependencies.back().srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependencies.back().srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies.back().dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies.back().dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    renderPass = utils::vkDefault::RenderPass(device, attachments, subpasses, dependencies);
}

void BloomGraphics::createFramebuffers(){
    framebuffers.resize(imageInfo.Count * (frames.size() + 1));
    for(size_t i = 0; i < frames.size(); i++){
        for (size_t j = 0; j < imageInfo.Count; j++){
            VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &frames[i].imageView(j);
                framebufferInfo.width = imageInfo.Extent.width;
                framebufferInfo.height = imageInfo.Extent.height;
                framebufferInfo.layers = 1;
            framebuffers[imageInfo.Count * i + j] = utils::vkDefault::Framebuffer(device, framebufferInfo);
        }
    }
    for(size_t i = 0; i < imageInfo.Count; i++){
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &bufferAttachment.imageView(i);
            framebufferInfo.width = imageInfo.Extent.width;
            framebufferInfo.height = imageInfo.Extent.height;
            framebufferInfo.layers = 1;
        framebuffers[imageInfo.Count * frames.size() + i] = utils::vkDefault::Framebuffer(device, framebufferInfo);
    }
}

void BloomGraphics::Filter::create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass)
{
    this->vertShaderPath = vertShaderPath;
    this->fragShaderPath = fragShaderPath;
    this->device = device;

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);

    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, vertShaderPath);
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, fragShaderPath);
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
        pushConstantRange.back().size = sizeof(BloomPushConst);
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
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);

    descriptorPool = utils::vkDefault::DescriptorPool(device, {&descriptorSetLayout }, imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageInfo.Count);
}

void BloomGraphics::Bloom::create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass)
{
    this->vertShaderPath = vertShaderPath;
    this->fragShaderPath = fragShaderPath;
    this->device = device;

    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), parameters.blitAttachmentsCount));
    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);

    uint32_t specializationData = parameters.blitAttachmentsCount;
    VkSpecializationMapEntry specializationMapEntry{};
    specializationMapEntry.constantID = 0;
    specializationMapEntry.offset = 0;
    specializationMapEntry.size = sizeof(uint32_t);
    VkSpecializationInfo specializationInfo;
    specializationInfo.mapEntryCount = 1;
    specializationInfo.pMapEntries = &specializationMapEntry;
    specializationInfo.dataSize = sizeof(specializationData);
    specializationInfo.pData = &specializationData;

    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, vertShaderPath);
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, fragShaderPath, specializationInfo);
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
        pushConstantRange.back().size = sizeof(BloomPushConst);
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
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);

    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageInfo.Count);
}

void BloomGraphics::create(moon::utils::AttachmentsDatabase& aDatabase)
{
    if(parameters.enable && !created){
        frames.resize(parameters.blitAttachmentsCount);
        moon::utils::createAttachments(physicalDevice, device, imageInfo, parameters.blitAttachmentsCount, frames.data(), VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, utils::vkDefault::sampler());
        moon::utils::createAttachments(physicalDevice, device, imageInfo, 1, &bufferAttachment, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, utils::vkDefault::sampler());

        aDatabase.addAttachmentData(parameters.out.bloom, parameters.enable, &bufferAttachment);

        createRenderPass();
        createFramebuffers();
        filter.create(shadersPath / "customFilter/customFilterVert.spv", shadersPath / "customFilter/customFilterFrag.spv", device, renderPass);
        bloom.create(shadersPath / "bloom/bloomVert.spv", shadersPath / "bloom/bloomFrag.spv", device, renderPass);

        created = true;
    }
}

void BloomGraphics::updateDescriptorSets(
    const moon::utils::BuffersDatabase&,
    const moon::utils::AttachmentsDatabase& aDatabase)
{
    if(!parameters.enable || !created) return;

    srcAttachment = aDatabase.get(parameters.in.bloom);

    for (uint32_t i = 0; i < imageInfo.Count; i++) {
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = bufferAttachment.imageView(i);
        imageInfo.sampler = bufferAttachment.sampler();

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
        descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites.back().dstSet = filter.descriptorSets[i];
        descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
        descriptorWrites.back().dstArrayElement = 0;
        descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites.back().descriptorCount = 1;
        descriptorWrites.back().pImageInfo = &imageInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    for (size_t image = 0; image < imageInfo.Count; image++)
    {
        std::vector<VkDescriptorImageInfo> blitImageInfo(parameters.blitAttachmentsCount);
        for(uint32_t i = 0, index = 0; i < blitImageInfo.size(); i++, index++){
            blitImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            blitImageInfo[index].imageView = frames[i].imageView(image);
            blitImageInfo[index].sampler = frames[i].sampler();
        }

        std::vector<VkWriteDescriptorSet> descriptorWrites;
            descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = bloom.descriptorSets[image];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = static_cast<uint32_t>(blitImageInfo.size());
            descriptorWrites.back().pImageInfo = blitImageInfo.data();
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void BloomGraphics::updateCommandBuffer(uint32_t frameNumber){
    if(!parameters.enable || !created) return;

    VkImageSubresourceRange ImageSubresourceRange{};
        ImageSubresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ImageSubresourceRange.baseMipLevel = 0;
        ImageSubresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        ImageSubresourceRange.baseArrayLayer = 0;
        ImageSubresourceRange.layerCount = 1;
    VkClearColorValue clearColorValue{};
        clearColorValue.uint32[0] = 0;
        clearColorValue.uint32[1] = 0;
        clearColorValue.uint32[2] = 0;
        clearColorValue.uint32[3] = 0;

    std::vector<VkImage> blitImages(frames.size());
    blitImages[0] = srcAttachment->image(frameNumber);
    moon::utils::texture::transitionLayout(commandBuffers[frameNumber], blitImages[0], parameters.inputImageLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);

    for(size_t i = 1; i < frames.size(); i++){
        blitImages[i] = frames[i - 1].image(frameNumber);
    }

    VkImage blitBufferImage = bufferAttachment.image(frameNumber);
    uint32_t width = imageInfo.Extent.width;
    uint32_t height = imageInfo.Extent.height;

    for(uint32_t k = 0; k < frames.size(); k++){
        moon::utils::texture::transitionLayout(commandBuffers[frameNumber], blitBufferImage, (k == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
        vkCmdClearColorImage(commandBuffers[frameNumber], blitBufferImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColorValue, 1, &ImageSubresourceRange);
        moon::utils::texture::blitDown(commandBuffers[frameNumber], blitImages[k], 0, blitBufferImage, 0, width, height, 0, 1, parameters.blitFactor);
        moon::utils::texture::transitionLayout(commandBuffers[frameNumber], blitBufferImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);

        render(commandBuffers[frameNumber], frames[k], frameNumber, k * imageInfo.Count + frameNumber, &filter);
    }
    for(uint32_t k = 0; k < frames.size(); k++){
        moon::utils::texture::transitionLayout(commandBuffers[frameNumber],frames[k].image(frameNumber), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
    }
    render(commandBuffers[frameNumber], bufferAttachment, frameNumber, static_cast<uint32_t>(frames.size()) * imageInfo.Count + frameNumber, &bloom);
    moon::utils::texture::transitionLayout(commandBuffers[frameNumber], blitBufferImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
}

void BloomGraphics::render(VkCommandBuffer commandBuffer, const moon::utils::Attachments& image, uint32_t frameNumber, uint32_t framebufferIndex, Workbody* worker)
{
    std::vector<VkClearValue> clearValues = { image.clearValue() };

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[framebufferIndex];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        BloomPushConst pushConst{ parameters.xSamplerStep, parameters.ySamplerStep, parameters.blitFactor};
        vkCmdPushConstants(commandBuffer, worker->pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(BloomPushConst), &pushConst);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, worker->pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, worker->pipelineLayout, 0, 1, &worker->descriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffer, 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);
}

}
