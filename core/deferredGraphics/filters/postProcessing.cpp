#include "postProcessing.h"
#include "operations.h"
#include "vkdefault.h"
#include "texture.h"

void postProcessingGraphics::setBlurAttachment(attachments *blurAttachment)
{
    this->blurAttachment = blurAttachment;
}
void postProcessingGraphics::setBlitAttachments(uint32_t blitAttachmentCount, attachments* blitAttachments, float blitFactor)
{
    postProcessing.blitFactor = blitFactor;
    postProcessing.blitAttachmentCount = blitAttachmentCount;
    this->blitAttachments = blitAttachments;
}
void postProcessingGraphics::setSSLRAttachment(attachments* sslrAttachment)
{
    this->sslrAttachment = sslrAttachment;
}
void postProcessingGraphics::setSSAOAttachment(attachments* ssaoAttachment)
{
    this->ssaoAttachment = ssaoAttachment;
}
void postProcessingGraphics::setLayersAttachment(attachments* layersAttachment)
{
    this->layersAttachment = layersAttachment;
}

void postProcessingGraphics::destroy()
{
    postProcessing.destroy(device);

    filterGraphics::destroy();
}

void postProcessingGraphics::destroySwapChainAttachments()
{
    for(size_t i=0; i<swapChainAttachments.size(); i++)
        for(size_t j=0; j <image.Count;j++)
            if(swapChainAttachments[i].imageView[j]) vkDestroyImageView(device,swapChainAttachments[i].imageView[j],nullptr);
    swapChainAttachments.resize(0);
}

void postProcessingGraphics::createSwapChain(VkSwapchainKHR* swapChain, GLFWwindow* window, SwapChain::SupportDetails* swapChainSupport, VkSurfaceKHR* surface, uint32_t queueFamilyIndexCount, uint32_t* pQueueFamilyIndices)
{
    VkPresentModeKHR presentMode = SwapChain::queryingPresentMode(swapChainSupport->presentModes);
    VkSurfaceFormatKHR surfaceFormat = SwapChain::queryingSurfaceFormat(swapChainSupport->formats);
    VkExtent2D extent = SwapChain::queryingExtent(window, swapChainSupport->capabilities);

    VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = *surface;
        createInfo.minImageCount = image.Count;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT ;
        createInfo.imageSharingMode = queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
        createInfo.pQueueFamilyIndices = pQueueFamilyIndices;
        createInfo.queueFamilyIndexCount = queueFamilyIndexCount;
        createInfo.preTransform = swapChainSupport->capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
    vkCreateSwapchainKHR(device, &createInfo, nullptr, swapChain);
}

void postProcessingGraphics::createSwapChainAttachments(VkSwapchainKHR* swapChain)
{
    swapChainAttachments.resize(swapChainAttachmentCount);
    for(size_t i=0;i<swapChainAttachments.size();i++)
    {
        swapChainAttachments[i].image.resize(image.Count);
        swapChainAttachments[i].imageView.resize(image.Count);
        vkGetSwapchainImagesKHR(device, *swapChain, &image.Count, swapChainAttachments[i].image.data());
    }

    for(size_t i=0;i<swapChainAttachments.size();i++)
        for (size_t size = 0; size < swapChainAttachments[i].imageView.size(); size++)
            Texture::createView(    device,
                                    VK_IMAGE_VIEW_TYPE_2D,
                                    image.Format,
                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                    1,
                                    0,
                                    1,
                                    swapChainAttachments[i].image[size],
                                    &swapChainAttachments[i].imageView[size]);
}

void postProcessingGraphics::createRenderPass()
{
    std::vector<VkAttachmentDescription> attachments = {
        attachments::imageDescription(image.Format, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
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

void postProcessingGraphics::createFramebuffers()
{
    framebuffers.resize(image.Count);
    for (size_t Image = 0; Image < framebuffers.size(); Image++)
    {
        std::vector<VkImageView> attachments(swapChainAttachments.size());
        for(size_t index=0; index<swapChainAttachments.size(); index++){
            attachments[index] = swapChainAttachments[index].imageView[Image];
        }

        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[Image]);
    }
}

void postProcessingGraphics::createPipelines()
{
    postProcessing.vertShaderPath = externalPath + "core\\deferredGraphics\\shaders\\postProcessing\\postProcessingVert.spv";
    postProcessing.fragShaderPath = externalPath + "core\\deferredGraphics\\shaders\\postProcessing\\postProcessingFrag.spv";
    postProcessing.createDescriptorSetLayout(device);
    postProcessing.createPipeline(device,&image,renderPass);
}

void postProcessingGraphics::PostProcessing::createDescriptorSetLayout(VkDevice device)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), blitAttachmentCount));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    VkDescriptorSetLayoutCreateInfo textureLayoutInfo{};
        textureLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        textureLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        textureLayoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device, &textureLayoutInfo, nullptr, &DescriptorSetLayout);
}

void postProcessingGraphics::PostProcessing::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass)
{
    uint32_t specializationData = blitAttachmentCount;
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
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    shaderStages.push_back(VkPipelineShaderStageCreateInfo{});
        shaderStages.back().sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages.back().stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages.back().module = vertShaderModule;
        shaderStages.back().pName = "main";
    shaderStages.push_back(VkPipelineShaderStageCreateInfo{});
        shaderStages.back().sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages.back().stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages.back().module = fragShaderModule;
        shaderStages.back().pName = "main";
        shaderStages.back().pSpecializationInfo = &specializationInfo;

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
        pushConstantRange.back().size = sizeof(postProcessingPushConst);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &DescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &PipelineLayout);

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
        pipelineInfo.back().layout = PipelineLayout;
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline);

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void postProcessingGraphics::createDescriptorPool(){
    filterGraphics::createDescriptorPool(device, &postProcessing, 0, (postProcessing.blitAttachmentCount + 4) * image.Count, postProcessing.blitAttachmentCount * image.Count);
}

void postProcessingGraphics::createDescriptorSets(){
    filterGraphics::createDescriptorSets(device, &postProcessing, image.Count);
}

void postProcessingGraphics::updateDescriptorSets()
{
    for (size_t image = 0; image < this->image.Count; image++)
    {
        VkDescriptorImageInfo layersImageInfo;
            layersImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            layersImageInfo.imageView = layersAttachment->imageView[image];
            layersImageInfo.sampler = layersAttachment->sampler;

        VkDescriptorImageInfo blurImageInfo;
            blurImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            blurImageInfo.imageView = blurAttachment ? blurAttachment->imageView[image] : *emptyTexture->getTextureImageView();
            blurImageInfo.sampler = blurAttachment ? blurAttachment->sampler : *emptyTexture->getTextureSampler();

        VkDescriptorImageInfo sslrImageInfo;
            sslrImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            sslrImageInfo.imageView = sslrAttachment ? sslrAttachment->imageView[image] : *emptyTexture->getTextureImageView();
            sslrImageInfo.sampler = sslrAttachment ? sslrAttachment->sampler : *emptyTexture->getTextureSampler();

        VkDescriptorImageInfo ssaoImageInfo;
            ssaoImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ssaoImageInfo.imageView = ssaoAttachment ? ssaoAttachment->imageView[image] : *emptyTexture->getTextureImageView();
            ssaoImageInfo.sampler = ssaoAttachment ? ssaoAttachment->sampler : *emptyTexture->getTextureSampler();

        std::vector<VkDescriptorImageInfo> blitImageInfo(postProcessing.blitAttachmentCount);
        for(uint32_t i = 0, index = 0; i < blitImageInfo.size(); i++, index++){
            blitImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            blitImageInfo[index].imageView = blitAttachments ? blitAttachments[i].imageView[image] : *emptyTexture->getTextureImageView();
            blitImageInfo[index].sampler = blitAttachments ? blitAttachments[i].sampler : *emptyTexture->getTextureSampler();
        }

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &layersImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &blurImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = static_cast<uint32_t>(blitImageInfo.size());
            descriptorWrites.back().pImageInfo = blitImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &sslrImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &ssaoImageInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void postProcessingGraphics::updateCommandBuffer(uint32_t frameNumber)
{
    std::vector<VkClearValue> clearValues;
    clearValues.push_back(VkClearValue{});

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        postProcessingPushConst pushConst{};
            pushConst.blitFactor = postProcessing.blitFactor;
        vkCmdPushConstants(commandBuffers[frameNumber], postProcessing.PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(postProcessingPushConst), &pushConst);

        vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, postProcessing.Pipeline);
        vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, postProcessing.PipelineLayout, 0, 1, &postProcessing.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers[frameNumber], 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}
