#include "bloom.h"
#include "operations.h"
#include "vkdefault.h"

bloomGraphics::bloomGraphics(bool enable, uint32_t blitAttachmentsCount, float blitFactor, float xSamplerStep, float ySamplerStep):
    enable(enable),
    blitFactor(blitFactor),
    xSamplerStep(xSamplerStep),
    ySamplerStep(ySamplerStep){
    bloom.blitAttachmentsCount = blitAttachmentsCount;
}

bloomGraphics& bloomGraphics::setBlitFactor(const float& blitFactor){this->blitFactor = blitFactor; return *this;}
bloomGraphics& bloomGraphics::setSamplerStepX(const float& xSamplerStep){this->xSamplerStep = xSamplerStep; return *this;}
bloomGraphics& bloomGraphics::setSamplerStepY(const float& ySamplerStep){this->ySamplerStep = ySamplerStep; return *this;}

void bloomGraphics::createAttachments(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap)
{
    frames.resize(bloom.blitAttachmentsCount);
    ::createAttachments(physicalDevice, device, image, bloom.blitAttachmentsCount, frames.data(), VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ::createAttachments(physicalDevice, device, image, 1, &bufferAttachment, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    attachmentsMap["bloomFinal"] = {enable, {&bufferAttachment}};
}

void bloomGraphics::destroy(){
    filter.destroy(device);
    bloom.destroy(device);
    workflow::destroy();

    for(auto& attachment: frames){
        attachment.deleteAttachment(device);
        attachment.deleteSampler(device);
    }

    bufferAttachment.deleteAttachment(device);
    bufferAttachment.deleteSampler(device);
}

void bloomGraphics::createRenderPass(){
    std::vector<VkAttachmentDescription> attachments = {
        attachments::imageDescription(image.Format,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    };

    std::vector<std::vector<VkAttachmentReference>> attachmentRef;
    attachmentRef.push_back(std::vector<VkAttachmentReference>());
        attachmentRef.back().push_back(VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    std::vector<VkSubpassDescription> subpass;
    for(auto refIt = attachmentRef.begin(); refIt != attachmentRef.end(); refIt++){
        subpass.push_back(VkSubpassDescription{});
            subpass.back().pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.back().colorAttachmentCount = static_cast<uint32_t>(refIt->size());
            subpass.back().pColorAttachments = refIt->data();
    }

    std::vector<VkSubpassDependency> dependency;
    dependency.push_back(VkSubpassDependency{});
        dependency.back().srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.back().dstSubpass = 0;
        dependency.back().srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency.back().srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependency.back().dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.back().dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = static_cast<uint32_t>(subpass.size());
        renderPassInfo.pSubpasses = subpass.data();
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependency.size());
        renderPassInfo.pDependencies = dependency.data();
    vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass);
}

void bloomGraphics::createFramebuffers(){
    framebuffers.resize(image.Count * (frames.size() + 1));
    for(size_t i = 0; i < frames.size(); i++){
        for (size_t j = 0; j < image.Count; j++){
            VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &frames[i].instances[j].imageView;
                framebufferInfo.width = image.frameBufferExtent.width;
                framebufferInfo.height = image.frameBufferExtent.height;
                framebufferInfo.layers = 1;
            vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[image.Count * i + j]);
        }
    }
    for(size_t i = 0; i < image.Count; i++){
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &bufferAttachment.instances[i].imageView;
            framebufferInfo.width = image.frameBufferExtent.width;
            framebufferInfo.height = image.frameBufferExtent.height;
            framebufferInfo.layers = 1;
        vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[image.Count * frames.size() + i]);
    }
}

void bloomGraphics::createPipelines(){
    filter.vertShaderPath = shadersPath / "customFilter/customFilterVert.spv";
    filter.fragShaderPath = shadersPath / "customFilter/customFilterFrag.spv";
    filter.createDescriptorSetLayout(device);
    filter.createPipeline(device,&image,renderPass);

    bloom.vertShaderPath = shadersPath / "bloom/bloomVert.spv";
    bloom.fragShaderPath = shadersPath / "bloom/bloomFrag.spv";
    bloom.createDescriptorSetLayout(device);
    bloom.createPipeline(device,&image,renderPass);
}

void bloomGraphics::Filter::createDescriptorSetLayout(VkDevice device){
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    VkDescriptorSetLayoutCreateInfo textureLayoutInfo{};
        textureLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        textureLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        textureLayoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device, &textureLayoutInfo, nullptr, &DescriptorSetLayout);
}

void bloomGraphics::Filter::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass)
{
    auto vertShaderCode = ShaderModule::readFile(vertShaderPath);
    auto fragShaderCode = ShaderModule::readFile(fragShaderPath);
    VkShaderModule vertShaderModule = ShaderModule::create(&device, vertShaderCode);
    VkShaderModule fragShaderModule = ShaderModule::create(&device, fragShaderCode);
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        vkDefault::vertrxShaderStage(vertShaderModule),
        vkDefault::fragmentShaderStage(fragShaderModule)
    };

    VkViewport viewport = vkDefault::viewport(pInfo->Offset, pInfo->Extent);
    VkRect2D scissor = vkDefault::scissor({0,0}, pInfo->frameBufferExtent);
    VkPipelineViewportStateCreateInfo viewportState = vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = vkDefault::depthStencilDisable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {vkDefault::colorBlendAttachmentState(VK_FALSE)};
    VkPipelineColorBlendStateCreateInfo colorBlending = vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(bloomPushConst);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &DescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
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

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void bloomGraphics::Bloom::createDescriptorSetLayout(VkDevice device){
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), blitAttachmentsCount));
    VkDescriptorSetLayoutCreateInfo textureLayoutInfo{};
    textureLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    textureLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    textureLayoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device, &textureLayoutInfo, nullptr, &DescriptorSetLayout);
}

void bloomGraphics::Bloom::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass)
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

    auto vertShaderCode = ShaderModule::readFile(vertShaderPath);
    auto fragShaderCode = ShaderModule::readFile(fragShaderPath);
    VkShaderModule vertShaderModule = ShaderModule::create(&device, vertShaderCode);
    VkShaderModule fragShaderModule = ShaderModule::create(&device, fragShaderCode);
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        vkDefault::vertrxShaderStage(vertShaderModule),
        vkDefault::fragmentShaderStage(fragShaderModule)
    };
    shaderStages.back().pSpecializationInfo = &specializationInfo;

    VkViewport viewport = vkDefault::viewport(pInfo->Offset, pInfo->Extent);
    VkRect2D scissor = vkDefault::scissor({0,0}, pInfo->frameBufferExtent);
    VkPipelineViewportStateCreateInfo viewportState = vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = vkDefault::depthStencilDisable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {vkDefault::colorBlendAttachmentState(VK_FALSE)};
    VkPipelineColorBlendStateCreateInfo colorBlending = vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
    pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
    pushConstantRange.back().offset = 0;
    pushConstantRange.back().size = sizeof(bloomPushConst);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &DescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
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

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void bloomGraphics::createDescriptorPool(){
    workflow::createDescriptorPool(device, &filter, 0, image.Count, image.Count);
    workflow::createDescriptorPool(device, &bloom, 0, bloom.blitAttachmentsCount * image.Count, bloom.blitAttachmentsCount * image.Count);
}

void bloomGraphics::createDescriptorSets(){
    workflow::createDescriptorSets(device, &filter, image.Count);
    workflow::createDescriptorSets(device, &bloom, image.Count);
}

void bloomGraphics::create(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap)
{
    if(enable){
        createAttachments(attachmentsMap);
        createRenderPass();
        createFramebuffers();
        createPipelines();
        createDescriptorPool();
        createDescriptorSets();
    }
}

void bloomGraphics::updateDescriptorSets(
    const std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>>&,
    const std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap)
{
    srcAttachment = attachmentsMap.at("combined.bloom").second.front();

    auto updateDescriptorSets = [](VkDevice device, attachments* image, VkSampler sampler, std::vector<VkDescriptorSet>& descriptorSets){
        auto imageIt = image->instances.begin();
        auto setIt = descriptorSets.begin();
        for (;imageIt != image->instances.end() && setIt != descriptorSets.end(); imageIt++, setIt++){
            VkDescriptorImageInfo imageInfo{};
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo.imageView = imageIt->imageView;
                imageInfo.sampler = sampler;

            std::vector<VkWriteDescriptorSet> descriptorWrites;
            descriptorWrites.push_back(VkWriteDescriptorSet{});
                descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites.back().dstSet = *setIt;
                descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
                descriptorWrites.back().dstArrayElement = 0;
                descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                descriptorWrites.back().descriptorCount = 1;
                descriptorWrites.back().pImageInfo = &imageInfo;
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    };
    updateDescriptorSets(device, &bufferAttachment, bufferAttachment.sampler, filter.DescriptorSets);

    for (size_t image = 0; image < this->image.Count; image++)
    {
        std::vector<VkDescriptorImageInfo> blitImageInfo(bloom.blitAttachmentsCount);
        for(uint32_t i = 0, index = 0; i < blitImageInfo.size(); i++, index++){
            blitImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            blitImageInfo[index].imageView = frames[i].instances[image].imageView;
            blitImageInfo[index].sampler = frames[i].sampler;
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

void bloomGraphics::updateCommandBuffer(uint32_t frameNumber)
{
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
    blitImages[0] = srcAttachment->instances[frameNumber].image;
    Texture::transitionLayout(commandBuffers[frameNumber], blitImages[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);

    for(size_t i=1;i<frames.size();i++){
        blitImages[i] = frames[i - 1].instances[frameNumber].image;
    }

    VkImage blitBufferImage = bufferAttachment.instances[frameNumber].image;
    uint32_t width = image.frameBufferExtent.width;
    uint32_t height = image.frameBufferExtent.height;

    for(uint32_t k = 0; k < frames.size(); k++){
        Texture::transitionLayout(commandBuffers[frameNumber], blitBufferImage, (k == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
        vkCmdClearColorImage(commandBuffers[frameNumber], blitBufferImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColorValue, 1, &ImageSubresourceRange);
        Texture::blitDown(commandBuffers[frameNumber], blitImages[k], 0, blitBufferImage, 0, width, height, 0, 1, blitFactor);
        Texture::transitionLayout(commandBuffers[frameNumber], blitBufferImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);

        render(commandBuffers[frameNumber], frames[k], frameNumber, k * image.Count + frameNumber, &filter);
    }
    for(uint32_t k = 0; k < frames.size(); k++){
        Texture::transitionLayout(commandBuffers[frameNumber],frames[k].instances[frameNumber].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
    }
    render(commandBuffers[frameNumber], bufferAttachment, frameNumber, frames.size() * image.Count + frameNumber, &bloom);
    Texture::transitionLayout(commandBuffers[frameNumber], blitBufferImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
}

void bloomGraphics::render(VkCommandBuffer commandBuffer, attachments image, uint32_t frameNumber, uint32_t framebufferIndex, workbody* worker)
{
    std::vector<VkClearValue> clearValues(1,VkClearValue{});
    for(uint32_t index = 0; index < clearValues.size(); index++){
        clearValues[index].color = image.clearValue.color;
    }

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[framebufferIndex];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = this->image.frameBufferExtent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        bloomPushConst pushConst{xSamplerStep, ySamplerStep, blitFactor};
        vkCmdPushConstants(commandBuffer, worker->PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(bloomPushConst), &pushConst);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, worker->Pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, worker->PipelineLayout, 0, 1, &worker->DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffer, 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);
}
