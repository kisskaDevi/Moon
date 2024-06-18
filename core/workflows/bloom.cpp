#include "bloom.h"
#include "operations.h"
#include "vkdefault.h"

namespace moon::workflows {

BloomGraphics::BloomGraphics(BloomParameters parameters, bool enable, uint32_t blitAttachmentsCount, VkImageLayout inputImageLayout, float blitFactor, float xSamplerStep, float ySamplerStep):
    parameters(parameters),
    enable(enable),
    blitFactor(blitFactor),
    xSamplerStep(xSamplerStep),
    ySamplerStep(ySamplerStep),
    inputImageLayout(inputImageLayout)
{
    bloom.blitAttachmentsCount = blitAttachmentsCount;
}

BloomGraphics& BloomGraphics::setBlitFactor(const float& blitFactor){this->blitFactor = blitFactor; return *this;}
BloomGraphics& BloomGraphics::setSamplerStepX(const float& xSamplerStep){this->xSamplerStep = xSamplerStep; return *this;}
BloomGraphics& BloomGraphics::setSamplerStepY(const float& ySamplerStep){this->ySamplerStep = ySamplerStep; return *this;}

void BloomGraphics::createAttachments(moon::utils::AttachmentsDatabase& aDatabase) {
    frames.resize(bloom.blitAttachmentsCount);
    moon::utils::createAttachments(physicalDevice, device, image, bloom.blitAttachmentsCount, frames.data(), VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, utils::vkDefault::samler());
    moon::utils::createAttachments(physicalDevice, device, image, 1, &bufferAttachment, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, utils::vkDefault::samler());

    aDatabase.addAttachmentData(parameters.out.bloom, enable, &bufferAttachment);
}

void BloomGraphics::destroy(){
    filter.destroy(device);
    bloom.destroy(device);
}

void BloomGraphics::createRenderPass(){
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        moon::utils::Attachments::imageDescription(image.Format,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
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

    CHECK(renderPass.create(device, attachments, subpasses, dependencies));
}

void BloomGraphics::createFramebuffers(){
    framebuffers.resize(image.Count * (frames.size() + 1));
    for(size_t i = 0; i < frames.size(); i++){
        for (size_t j = 0; j < image.Count; j++){
            VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &frames[i].imageView(j);
                framebufferInfo.width = image.Extent.width;
                framebufferInfo.height = image.Extent.height;
                framebufferInfo.layers = 1;
            CHECK(framebuffers[image.Count * i + j].create(device, framebufferInfo));
        }
    }
    for(size_t i = 0; i < image.Count; i++){
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &bufferAttachment.imageView(i);
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        CHECK(framebuffers[image.Count * frames.size() + i].create(device, framebufferInfo));
    }
}

void BloomGraphics::createPipelines(){
    filter.vertShaderPath = shadersPath / "customFilter/customFilterVert.spv";
    filter.fragShaderPath = shadersPath / "customFilter/customFilterFrag.spv";
    filter.createDescriptorSetLayout(device);
    filter.createPipeline(device,&image,renderPass);

    bloom.vertShaderPath = shadersPath / "bloom/bloomVert.spv";
    bloom.fragShaderPath = shadersPath / "bloom/bloomFrag.spv";
    bloom.createDescriptorSetLayout(device);
    bloom.createPipeline(device,&image,renderPass);
}

void BloomGraphics::Filter::createDescriptorSetLayout(VkDevice device){
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    CHECK(descriptorSetLayout.create(device, bindings));
}

void BloomGraphics::Filter::createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass)
{
    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, vertShaderPath);
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, fragShaderPath);
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
        pushConstantRange.back().size = sizeof(BloomPushConst);
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
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    CHECK(pipeline.create(device, pipelineInfo));
}

void BloomGraphics::Bloom::createDescriptorSetLayout(VkDevice device){
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), blitAttachmentsCount));

    CHECK(descriptorSetLayout.create(device, bindings));
}

void BloomGraphics::Bloom::createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass)
{
    uint32_t specializationData = blitAttachmentsCount;
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
        pushConstantRange.back().size = sizeof(BloomPushConst);
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
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    CHECK(pipeline.create(device, pipelineInfo));
}

void BloomGraphics::createDescriptorPool(){
    Workflow::createDescriptorPool(device, &filter, 0, image.Count, image.Count);
    Workflow::createDescriptorPool(device, &bloom, 0, bloom.blitAttachmentsCount * image.Count, bloom.blitAttachmentsCount * image.Count);
}

void BloomGraphics::createDescriptorSets(){
    Workflow::createDescriptorSets(device, &filter, image.Count);
    Workflow::createDescriptorSets(device, &bloom, image.Count);
}

void BloomGraphics::create(moon::utils::AttachmentsDatabase& aDatabase)
{
    if(enable){
        createAttachments(aDatabase);
        createRenderPass();
        createFramebuffers();
        createPipelines();
        createDescriptorPool();
        createDescriptorSets();
    }
}

void BloomGraphics::updateDescriptorSets(
    const moon::utils::BuffersDatabase&,
    const moon::utils::AttachmentsDatabase& aDatabase)
{
    if(!enable) return;

    srcAttachment = aDatabase.get(parameters.in.bloom);

    auto updateDescriptorSets = [](VkDevice device, const moon::utils::Attachments& image, const std::vector<VkDescriptorSet>& descriptorSets){
        for (uint32_t i = 0; i < image.count(); i++){
            VkDescriptorImageInfo imageInfo{};
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo.imageView = image.imageView(i);
                imageInfo.sampler = image.sampler();

            std::vector<VkWriteDescriptorSet> descriptorWrites;
            descriptorWrites.push_back(VkWriteDescriptorSet{});
                descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites.back().dstSet = descriptorSets[i];
                descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
                descriptorWrites.back().dstArrayElement = 0;
                descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                descriptorWrites.back().descriptorCount = 1;
                descriptorWrites.back().pImageInfo = &imageInfo;
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    };
    updateDescriptorSets(device, bufferAttachment, filter.DescriptorSets);

    for (size_t image = 0; image < this->image.Count; image++)
    {
        std::vector<VkDescriptorImageInfo> blitImageInfo(bloom.blitAttachmentsCount);
        for(uint32_t i = 0, index = 0; i < blitImageInfo.size(); i++, index++){
            blitImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            blitImageInfo[index].imageView = frames[i].imageView(image);
            blitImageInfo[index].sampler = frames[i].sampler();
        }

        std::vector<VkWriteDescriptorSet> descriptorWrites;
            descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = bloom.DescriptorSets[image];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = static_cast<uint32_t>(blitImageInfo.size());
            descriptorWrites.back().pImageInfo = blitImageInfo.data();
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void BloomGraphics::updateCommandBuffer(uint32_t frameNumber){
    if(!enable) return;

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
    moon::utils::texture::transitionLayout(commandBuffers[frameNumber], blitImages[0], inputImageLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);

    for(size_t i=1;i<frames.size();i++){
        blitImages[i] = frames[i - 1].image(frameNumber);
    }

    VkImage blitBufferImage = bufferAttachment.image(frameNumber);
    uint32_t width = image.Extent.width;
    uint32_t height = image.Extent.height;

    for(uint32_t k = 0; k < frames.size(); k++){
        moon::utils::texture::transitionLayout(commandBuffers[frameNumber], blitBufferImage, (k == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
        vkCmdClearColorImage(commandBuffers[frameNumber], blitBufferImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColorValue, 1, &ImageSubresourceRange);
        moon::utils::texture::blitDown(commandBuffers[frameNumber], blitImages[k], 0, blitBufferImage, 0, width, height, 0, 1, blitFactor);
        moon::utils::texture::transitionLayout(commandBuffers[frameNumber], blitBufferImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);

        render(commandBuffers[frameNumber], frames[k], frameNumber, k * image.Count + frameNumber, &filter);
    }
    for(uint32_t k = 0; k < frames.size(); k++){
        moon::utils::texture::transitionLayout(commandBuffers[frameNumber],frames[k].image(frameNumber), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
    }
    render(commandBuffers[frameNumber], bufferAttachment, frameNumber, static_cast<uint32_t>(frames.size()) * image.Count + frameNumber, &bloom);
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
        renderPassInfo.renderArea.extent = this->image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        BloomPushConst pushConst{xSamplerStep, ySamplerStep, blitFactor};
        vkCmdPushConstants(commandBuffer, worker->pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(BloomPushConst), &pushConst);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, worker->pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, worker->pipelineLayout, 0, 1, &worker->DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffer, 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);
}

}
