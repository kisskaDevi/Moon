#include "customfilter.h"
#include "core/operations.h"
#include "../bufferObjects.h"

#include <iostream>
#include <array>

customFilter::customFilter()
{

}

void customFilter::setExternalPath(const std::string& path)
{
    filter.ExternalPath = path;
}

void customFilter::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool)
{
    this->physicalDevice = physicalDevice;
    this->device = device;
    this->graphicsQueue = graphicsQueue;
    this->commandPool = commandPool;
}
void customFilter::setImageProp(imageInfo* pInfo)                           {this->image = *pInfo;}
void customFilter::setSampleStep(float deltaX, float deltaY)                {xSampleStep = deltaX; ySampleStep = deltaY;}
void customFilter::setAttachments(uint32_t attachmentsCount, attachments* pAttachments)
{
    this->attachmentsCount = attachmentsCount;
    this->pAttachments = pAttachments;
}

void customFilter::setSrcAttachment(attachments *srcAttachment)
{
    this->srcAttachment = srcAttachment;
}

void customFilter::setBlitFactor(const float &blitFactor)
{
    this->blitFactor = blitFactor;
}

void customFilter::createBufferAttachments()
{
    bufferAttachment.resize(image.Count);
    for(size_t imageNumber=0; imageNumber<image.Count; imageNumber++)
    {
        createImage(            physicalDevice,
                                device,
                                image.Extent.width,
                                image.Extent.height,
                                1,
                                VK_SAMPLE_COUNT_1_BIT,
                                image.Format,
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT ,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                bufferAttachment.image[imageNumber],
                                bufferAttachment.imageMemory[imageNumber]);

        bufferAttachment.imageView[imageNumber] =
        createImageView(        device,
                                bufferAttachment.image[imageNumber],
                                image.Format,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1);

        transitionImageLayout(  device,
                                graphicsQueue,
                                commandPool,
                                bufferAttachment.image[imageNumber],
                                VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_REMAINING_MIP_LEVELS);
    }
    VkSamplerCreateInfo SamplerInfo{};
        SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        SamplerInfo.magFilter = VK_FILTER_LINEAR;
        SamplerInfo.minFilter = VK_FILTER_LINEAR;
        SamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        SamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        SamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        SamplerInfo.anisotropyEnable = VK_TRUE;
        SamplerInfo.maxAnisotropy = 1.0f;
        SamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        SamplerInfo.unnormalizedCoordinates = VK_FALSE;
        SamplerInfo.compareEnable = VK_FALSE;
        SamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        SamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        SamplerInfo.minLod = 0.0f;
        SamplerInfo.maxLod = 0.0f;
        SamplerInfo.mipLodBias = 0.0f;
    vkCreateSampler(*device, &SamplerInfo, nullptr, &bufferAttachment.sampler);
}

void customFilter::createAttachments(uint32_t attachmentsCount, attachments* pAttachments)
{
    for(uint32_t i=0;i<attachmentsCount;i++){
        pAttachments[i].resize(image.Count);
        for(size_t imageNumber=0; imageNumber<image.Count; imageNumber++)
        {
            createImage(        physicalDevice,
                                device,
                                image.Extent.width,
                                image.Extent.height,
                                1,
                                VK_SAMPLE_COUNT_1_BIT,
                                image.Format,
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                pAttachments[i].image[imageNumber],
                                pAttachments[i].imageMemory[imageNumber]);

            pAttachments[i].imageView[imageNumber] =
            createImageView(    device,
                                pAttachments[i].image[imageNumber],
                                image.Format,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1);
        }
        VkSamplerCreateInfo SamplerInfo{};
            SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            SamplerInfo.magFilter = VK_FILTER_LINEAR;
            SamplerInfo.minFilter = VK_FILTER_LINEAR;
            SamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            SamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            SamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            SamplerInfo.anisotropyEnable = VK_TRUE;
            SamplerInfo.maxAnisotropy = 1.0f;
            SamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
            SamplerInfo.unnormalizedCoordinates = VK_FALSE;
            SamplerInfo.compareEnable = VK_FALSE;
            SamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
            SamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            SamplerInfo.minLod = 0.0f;
            SamplerInfo.maxLod = 0.0f;
            SamplerInfo.mipLodBias = 0.0f;
        vkCreateSampler(*device, &SamplerInfo, nullptr, &pAttachments[i].sampler);
    }
}

void customFilter::Filter::Destroy(VkDevice* device)
{
    if(Pipeline)            vkDestroyPipeline(*device, Pipeline, nullptr);
    if(PipelineLayout)      vkDestroyPipelineLayout(*device, PipelineLayout,nullptr);
    if(DescriptorSetLayout) vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr);
    if(DescriptorPool)      vkDestroyDescriptorPool(*device, DescriptorPool, nullptr);
}

void customFilter::destroy()
{
    filter.Destroy(device);

    if(renderPass) vkDestroyRenderPass(*device, renderPass, nullptr);
    for(size_t i = 0; i< framebuffers.size();i++)
        for(size_t j = 0; j< framebuffers[i].size();j++)
            if(framebuffers[i][j]) vkDestroyFramebuffer(*device, framebuffers[i][j],nullptr);

    bufferAttachment.deleteAttachment(&*device);
    bufferAttachment.deleteSampler(&*device);
}

void customFilter::createRenderPass()
{
    uint32_t index = 0;
    std::array<VkAttachmentDescription,1> attachments{};
        attachments[index].format = image.Format;
        attachments[index].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[index].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[index].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[index].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[index].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[index].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[index].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    index = 0;
    std::array<VkAttachmentReference,1> attachmentRef;
        attachmentRef[index].attachment = 0;
        attachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    index = 0;
    std::array<VkSubpassDescription,1> subpass{};
        subpass[index].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass[index].colorAttachmentCount = static_cast<uint32_t>(attachmentRef.size());
        subpass[index].pColorAttachments = attachmentRef.data();

    index = 0;
    std::array<VkSubpassDependency,1> dependency{};                                                                                        
        dependency[index].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency[index].dstSubpass = 0;
        dependency[index].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency[index].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependency[index].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency[index].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = static_cast<uint32_t>(subpass.size());
        renderPassInfo.pSubpasses = subpass.data();
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependency.size());
        renderPassInfo.pDependencies = dependency.data();
    vkCreateRenderPass(*device, &renderPassInfo, nullptr, &renderPass);
}

void customFilter::createFramebuffers()
{
    framebuffers.resize(attachmentsCount);
    for(size_t i = 0; i < attachmentsCount; i++){
        framebuffers[i].resize(image.Count);
        for (size_t j = 0; j < framebuffers[i].size(); j++){
            VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &pAttachments[i].imageView[j];
                framebufferInfo.width = image.Extent.width;
                framebufferInfo.height = image.Extent.height;
                framebufferInfo.layers = 1;
            vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &framebuffers[i][j]);
        }
    }
}

void customFilter::createPipelines()
{
    filter.createDescriptorSetLayout(device);
    filter.createPipeline(device,&image,&renderPass);
}

void customFilter::Filter::createDescriptorSetLayout(VkDevice* device)
{
    uint32_t index = 0;

    std::array<VkDescriptorSetLayoutBinding,1> bindings{};
        bindings[index].binding = 0;
        bindings[index].descriptorCount = 1;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[index].pImmutableSamplers = nullptr;
        bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo textureLayoutInfo{};
        textureLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        textureLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        textureLayoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(*device, &textureLayoutInfo, nullptr, &DescriptorSetLayout);
}

void customFilter::Filter::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\customFilter\\customFilterVert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\customFilter\\customFilterFrag.spv");
    VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);
    std::array<VkPipelineShaderStageCreateInfo,2> shaderStages{};
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[index].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[index].module = vertShaderModule;
        shaderStages[index].pName = "main";
    index++;
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[index].module = fragShaderModule;
        shaderStages[index].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

    index = 0;
    std::array<VkViewport,1> viewport{};
        viewport[index].x = 0.0f;
        viewport[index].y = 0.0f;
        viewport[index].width  = (float) pInfo->Extent.width;
        viewport[index].height = (float) pInfo->Extent.height;
        viewport[index].minDepth = 0.0f;
        viewport[index].maxDepth = 1.0f;
    std::array<VkRect2D,1> scissor{};
        scissor[index].offset = {0, 0};
        scissor[index].extent = pInfo->Extent;
    VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = static_cast<uint32_t>(viewport.size());;
        viewportState.pViewports = viewport.data();
        viewportState.scissorCount = static_cast<uint32_t>(scissor.size());;
        viewportState.pScissors = scissor.data();

    VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

    index = 0;
    std::array<VkPipelineColorBlendAttachmentState,1> colorBlendAttachment;
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_FALSE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].alphaBlendOp = VK_BLEND_OP_MAX;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachment.size());
        colorBlending.pAttachments = colorBlendAttachment.data();
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};

    index=0;
    std::array<VkPushConstantRange,1> pushConstantRange{};
        pushConstantRange[index].stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange[index].offset = 0;
        pushConstantRange[index].size = sizeof(CustomFilterPushConst);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &DescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &PipelineLayout);

    VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = PipelineLayout;
        pipelineInfo.renderPass = *pRenderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.pDepthStencilState = &depthStencil;
    vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Pipeline);

    vkDestroyShaderModule(*device, fragShaderModule, nullptr);
    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
}

void customFilter::createDescriptorPool()
{
    size_t index = 0;
    std::array<VkDescriptorPoolSize,1> poolSizes;
    for(uint32_t i=0;i<poolSizes.size();i++,index++){
        poolSizes[index].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[index].descriptorCount = static_cast<uint32_t>(image.Count);
    }
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    vkCreateDescriptorPool(*device, &poolInfo, nullptr, &filter.DescriptorPool);
}

void customFilter::createDescriptorSets()
{
    filter.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, filter.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = filter.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(*device, &allocInfo, filter.DescriptorSets.data());
}

void customFilter::updateDescriptorSets()
{
    for (size_t i = 0; i < image.Count; i++)
    {
        uint32_t index = 0;
        std::array<VkDescriptorImageInfo, 1> imageInfo;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = bufferAttachment.imageView[i];
            imageInfo[index].sampler = bufferAttachment.sampler;

        index = 0;
        std::array<VkWriteDescriptorSet, 1> descriptorWrites{};
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = filter.DescriptorSets[i];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = static_cast<uint32_t>(imageInfo.size());
            descriptorWrites[index].pImageInfo = imageInfo.data();
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void customFilter::render(uint32_t frameNumber, VkCommandBuffer commandBuffer, uint32_t attachmentNumber)
{
    std::array<VkClearValue, 1> ClearValues{};
        ClearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[attachmentNumber][frameNumber];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(ClearValues.size());
        renderPassInfo.pClearValues = ClearValues.data();

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

void customFilter::render(uint32_t frameNumber, VkCommandBuffer commandBuffer)
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
        transitionImageLayout(&commandBuffer,blitBufferImage,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_REMAINING_MIP_LEVELS);
        vkCmdClearColorImage(commandBuffer,blitBufferImage,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ,&clearColorValue,1,&ImageSubresourceRange);
        blitDown(&commandBuffer,blitImages[k],blitBufferImage,width,height,blitFactor);
        transitionImageLayout(&commandBuffer,blitBufferImage,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_REMAINING_MIP_LEVELS);
        render(frameNumber,commandBuffer,k);
    }
    for(uint32_t k=0;k<attachmentsCount;k++)
        transitionImageLayout(&commandBuffer,pAttachments[k].image[frameNumber],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_REMAINING_MIP_LEVELS);
}