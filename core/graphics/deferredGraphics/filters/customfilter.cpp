#include "customfilter.h"
#include "core/operations.h"
#include "../vkdefault.h"

customFilter::customFilter()
{}

void customFilter::setSampleStep(const float& deltaX, const float& deltaY){
    xSampleStep = deltaX; ySampleStep = deltaY;
}

void customFilter::setSrcAttachment(attachments *srcAttachment){
    this->srcAttachment = srcAttachment;
}

void customFilter::setBlitFactor(const float &blitFactor){
    this->blitFactor = blitFactor;
}

void customFilter::createBufferAttachments(){
    bufferAttachment.create(&physicalDevice,&device,image.Format,VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,image.Extent,image.Count);
    VkSamplerCreateInfo SamplerInfo = vkDefault::samler();
    vkCreateSampler(device, &SamplerInfo, nullptr, &bufferAttachment.sampler);
}

void customFilter::createAttachments(uint32_t attachmentsCount, attachments* pAttachments){
    for(size_t index=0; index<attachmentsCount; index++){
        pAttachments[index].create(&physicalDevice,&device,image.Format,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,image.Extent,image.Count);
        VkSamplerCreateInfo SamplerInfo = vkDefault::samler();
        vkCreateSampler(device, &SamplerInfo, nullptr, &pAttachments[index].sampler);
    }
}

void customFilter::destroy(){
    filter.destroy(device);

    if(renderPass) {vkDestroyRenderPass(device, renderPass, nullptr); renderPass = VK_NULL_HANDLE;}
    for(size_t i = 0; i< framebuffers.size();i++)
        if(framebuffers[i]) vkDestroyFramebuffer(device, framebuffers[i],nullptr);
    framebuffers.resize(0);

    bufferAttachment.deleteAttachment(&device);
    bufferAttachment.deleteSampler(&device);
}

void customFilter::createRenderPass(){
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

void customFilter::createFramebuffers(){
    framebuffers.resize(image.Count*attachmentsCount);
    for(size_t i = 0; i < attachmentsCount; i++){
        for (size_t j = 0; j < image.Count; j++){
            VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &pAttachments[i].imageView[j];
                framebufferInfo.width = image.Extent.width;
                framebufferInfo.height = image.Extent.height;
                framebufferInfo.layers = 1;
            vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[image.Count * i + j]);
        }
    }
}

void customFilter::createPipelines(){
    filter.vertShaderPath = externalPath + "core\\graphics\\deferredGraphics\\shaders\\customFilter\\customFilterVert.spv";
    filter.fragShaderPath = externalPath + "core\\graphics\\deferredGraphics\\shaders\\customFilter\\customFilterFrag.spv";
    filter.createDescriptorSetLayout(device);
    filter.createPipeline(device,&image,renderPass);
}

void customFilter::Filter::createDescriptorSetLayout(VkDevice device){
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    VkDescriptorSetLayoutCreateInfo textureLayoutInfo{};
        textureLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        textureLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        textureLayoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device, &textureLayoutInfo, nullptr, &DescriptorSetLayout);
}

void customFilter::Filter::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass)
{
    auto vertShaderCode = ShaderModule::readFile(vertShaderPath);
    auto fragShaderCode = ShaderModule::readFile(fragShaderPath);
    VkShaderModule vertShaderModule = ShaderModule::create(&device, vertShaderCode);
    VkShaderModule fragShaderModule = ShaderModule::create(&device, fragShaderCode);
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        vkDefault::vertrxShaderStage(vertShaderModule),
        vkDefault::fragmentShaderStage(fragShaderModule)
    };

    VkViewport viewport = vkDefault::viewport(pInfo->Extent);
    VkRect2D scissor = vkDefault::scissor(pInfo->Extent);
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
        pushConstantRange.back().size = sizeof(CustomFilterPushConst);
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

void customFilter::createDescriptorPool(){
    filterGraphics::createDescriptorPool(device, &filter, 0, image.Count, image.Count);
}

void customFilter::createDescriptorSets(){
    filterGraphics::createDescriptorSets(device, &filter, image.Count);
}

void customFilter::updateDescriptorSets()
{
    auto updateDescriptorSets = [](VkDevice device, const std::vector<VkImageView>& imageViews, VkSampler sampler, std::vector<VkDescriptorSet>& descriptorSets){
        auto imageIt = imageViews.begin();
        auto setIt = descriptorSets.begin();
        for (;imageIt != imageViews.end() && setIt != descriptorSets.end(); imageIt++, setIt++){
            VkDescriptorImageInfo imageInfo{};
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo.imageView = *imageIt;
                imageInfo.sampler = sampler;

            std::vector<VkWriteDescriptorSet> descriptorWrites;
            descriptorWrites.push_back(VkWriteDescriptorSet{});
                descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites.back().dstSet = *setIt;
                descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
                descriptorWrites.back().dstArrayElement = 0;
                descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                descriptorWrites.back().descriptorCount = 1;
                descriptorWrites.back().pImageInfo = &imageInfo;
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    };
    updateDescriptorSets(device, bufferAttachment.imageView, bufferAttachment.sampler, filter.DescriptorSets);
}

void customFilter::updateCommandBuffer(uint32_t frameNumber)
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

        std::vector<VkImage> blitImages(attachmentsCount);
        blitImages[0] = srcAttachment->image[frameNumber];
        for(size_t i=1;i<attachmentsCount;i++){
            blitImages[i] = pAttachments[i-1].image[frameNumber];
        }
        VkImage blitBufferImage = bufferAttachment.image[frameNumber];
        uint32_t width = image.Extent.width;
        uint32_t height = image.Extent.height;

        for(uint32_t k=0;k<attachmentsCount;k++){
            Texture::transitionLayout(commandBuffers[frameNumber],blitBufferImage, k == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_REMAINING_MIP_LEVELS, 0, 1);
            vkCmdClearColorImage(commandBuffers[frameNumber],blitBufferImage,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ,&clearColorValue,1,&ImageSubresourceRange);
            Texture::blitDown(commandBuffers[frameNumber],blitImages[k],0,blitBufferImage,0,width,height,0,1,blitFactor);
            Texture::transitionLayout(commandBuffers[frameNumber],blitBufferImage,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_REMAINING_MIP_LEVELS, 0, 1);
            render(frameNumber,commandBuffers[frameNumber],k);
        }
        for(uint32_t k=0;k<attachmentsCount;k++){
            Texture::transitionLayout(commandBuffers[frameNumber],pAttachments[k].image[frameNumber],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_REMAINING_MIP_LEVELS, 0, 1);
        }
}

void customFilter::render(uint32_t frameNumber, VkCommandBuffer commandBuffer, uint32_t attachmentNumber)
{
    std::vector<VkClearValue> clearValues(1,VkClearValue{});
    for(uint32_t index = 0; index < clearValues.size(); index++){
        clearValues[index].color = pAttachments[attachmentNumber].clearValue.color;
    }

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[attachmentNumber * image.Count + frameNumber];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        CustomFilterPushConst pushConst{};
            pushConst.deltax = xSampleStep;
            pushConst.deltay = ySampleStep;
        vkCmdPushConstants(commandBuffer, filter.PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(CustomFilterPushConst), &pushConst);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, filter.Pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, filter.PipelineLayout, 0, 1, &filter.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffer, 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);
}
