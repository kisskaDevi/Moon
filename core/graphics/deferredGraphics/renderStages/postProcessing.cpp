#include "postProcessing.h"

#include <iostream>
#include <cstdint>          // UINT32_MAX
#include <array>
#include <algorithm>        // std::min/std::max

postProcessingGraphics::postProcessingGraphics()
{}

void postProcessingGraphics::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool, uint32_t graphicsFamily, uint32_t presentFamily)
{
    this->physicalDevice = physicalDevice;
    this->device = device;
    this->graphicsQueue = graphicsQueue;
    this->commandPool = commandPool;
    this->queueFamilyIndices = {graphicsFamily,presentFamily};
}
void postProcessingGraphics::setImageProp(imageInfo* pInfo)
{
    this->image = *pInfo;
}
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

void postProcessingGraphics::PostProcessing::Destroy(VkDevice* device)
{
    if(Pipeline)            {vkDestroyPipeline(*device, Pipeline, nullptr); Pipeline = VK_NULL_HANDLE;}
    if(PipelineLayout)      {vkDestroyPipelineLayout(*device, PipelineLayout,nullptr); PipelineLayout = VK_NULL_HANDLE;}
    if(DescriptorSetLayout) {vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr); DescriptorSetLayout = VK_NULL_HANDLE;}
    if(DescriptorPool)      {vkDestroyDescriptorPool(*device, DescriptorPool, nullptr); DescriptorPool = VK_NULL_HANDLE;}
}

void postProcessingGraphics::destroy()
{
    postProcessing.Destroy(device);

    if(renderPass) {vkDestroyRenderPass(*device, renderPass, nullptr); renderPass = VK_NULL_HANDLE;}
    for(size_t i = 0; i< framebuffers.size();i++)
        if(framebuffers[i]) vkDestroyFramebuffer(*device, framebuffers[i],nullptr);
    framebuffers.resize(0);
}

void postProcessingGraphics::destroySwapChainAttachments()
{
    for(size_t i=0; i<swapChainAttachments.size(); i++)
        for(size_t j=0; j <image.Count;j++)
            if(swapChainAttachments[i].imageView[j]) vkDestroyImageView(*device,swapChainAttachments[i].imageView[j],nullptr);
    swapChainAttachments.resize(0);
}

void postProcessingGraphics::setExternalPath(const std::string &path)
{
    postProcessing.ExternalPath = path;
}

void postProcessingGraphics::createSwapChain(VkSwapchainKHR* swapChain, GLFWwindow* window, SwapChainSupportDetails swapChainSupport, VkSurfaceKHR* surface)
{
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkExtent2D extent = chooseSwapExtent(window, swapChainSupport.capabilities);

    QueueFamilyIndices indices = queueFamilyIndices;
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = *surface;
        createInfo.minImageCount = image.Count;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT ;
        if(indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
            createInfo.queueFamilyIndexCount = 2;
        }else{
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
    vkCreateSwapchainKHR(*device, &createInfo, nullptr, swapChain);
}

void postProcessingGraphics::createSwapChainAttachments(VkSwapchainKHR* swapChain)
{
    swapChainAttachments.resize(swapChainAttachmentCount);
    for(size_t i=0;i<swapChainAttachments.size();i++)
    {
        swapChainAttachments[i].image.resize(image.Count);
        swapChainAttachments[i].imageView.resize(image.Count);
        vkGetSwapchainImagesKHR(*device, *swapChain, &image.Count, swapChainAttachments[i].image.data());
    }

    for(size_t i=0;i<swapChainAttachments.size();i++)
        for (size_t size = 0; size < swapChainAttachments[i].imageView.size(); size++)
            swapChainAttachments[i].imageView[size] =
            createImageView(    device,
                                swapChainAttachments[i].image[size],
                                image.Format,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1);
}

void postProcessingGraphics::createRenderPass()
{
    uint32_t index = 0;
    std::array<VkAttachmentDescription,1> attachments{};
    for(uint32_t index = 0; index < attachments.size(); index++)
        attachments[index] = attachments::imageDescription(image.Format,VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

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
        vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &framebuffers[Image]);
    }
}

void postProcessingGraphics::createPipelines()
{
    postProcessing.createDescriptorSetLayout(device);
    postProcessing.createPipeline(device,&image,&renderPass);
}
    void postProcessingGraphics::PostProcessing::createDescriptorSetLayout(VkDevice* device)
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings{};
        bindings.push_back(VkDescriptorSetLayoutBinding{});
            bindings.back().binding = bindings.size() - 1;
            bindings.back().descriptorCount = 1;
            bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings.back().pImmutableSamplers = nullptr;
            bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(VkDescriptorSetLayoutBinding{});
            bindings.back().binding = bindings.size() - 1;
            bindings.back().descriptorCount = 1;
            bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings.back().pImmutableSamplers = nullptr;
            bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(VkDescriptorSetLayoutBinding{});
            bindings.back().binding = bindings.size() - 1;
            bindings.back().descriptorCount = static_cast<uint32_t>(blitAttachmentCount);
            bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings.back().pImmutableSamplers = nullptr;
            bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(VkDescriptorSetLayoutBinding{});
            bindings.back().binding = bindings.size() - 1;
            bindings.back().descriptorCount = 1;
            bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings.back().pImmutableSamplers = nullptr;
            bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(VkDescriptorSetLayoutBinding{});
            bindings.back().binding = bindings.size() - 1;
            bindings.back().descriptorCount = 1;
            bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings.back().pImmutableSamplers = nullptr;
            bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo textureLayoutInfo{};
            textureLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            textureLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            textureLayoutInfo.pBindings = bindings.data();
        vkCreateDescriptorSetLayout(*device, &textureLayoutInfo, nullptr, &DescriptorSetLayout);
    }
    void postProcessingGraphics::PostProcessing::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
    {
        uint32_t index = 0;

        auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\postProcessing\\postProcessingVert.spv");
        auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\postProcessing\\postProcessingFrag.spv");
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
            viewport[index].height= (float) pInfo->Extent.height;
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
            pushConstantRange[index].size = sizeof(postProcessingPushConst);
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &DescriptorSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
        vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &PipelineLayout);

        index = 0;
        std::array<VkGraphicsPipelineCreateInfo,1> pipelineInfo{};
            pipelineInfo[index].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo[index].stageCount = static_cast<uint32_t>(shaderStages.size());
            pipelineInfo[index].pStages = shaderStages.data();
            pipelineInfo[index].pVertexInputState = &vertexInputInfo;
            pipelineInfo[index].pInputAssemblyState = &inputAssembly;
            pipelineInfo[index].pViewportState = &viewportState;
            pipelineInfo[index].pRasterizationState = &rasterizer;
            pipelineInfo[index].pMultisampleState = &multisampling;
            pipelineInfo[index].pColorBlendState = &colorBlending;
            pipelineInfo[index].layout = PipelineLayout;
            pipelineInfo[index].renderPass = *pRenderPass;
            pipelineInfo[index].subpass = 0;
            pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
            pipelineInfo[index].pDepthStencilState = &depthStencil;
        vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline);

        vkDestroyShaderModule(*device, fragShaderModule, nullptr);
        vkDestroyShaderModule(*device, vertShaderModule, nullptr);
    }

void postProcessingGraphics::createDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(postProcessing.blitAttachmentCount*image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    vkCreateDescriptorPool(*device, &poolInfo, nullptr, &postProcessing.DescriptorPool);
}

void postProcessingGraphics::createDescriptorSets()
{
    postProcessing.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, postProcessing.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = postProcessing.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(*device, &allocInfo, postProcessing.DescriptorSets.data());
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
            blurImageInfo.imageView = blurAttachment->imageView[image];
            blurImageInfo.sampler = blurAttachment->sampler;

        VkDescriptorImageInfo sslrImageInfo;
            sslrImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            sslrImageInfo.imageView = sslrAttachment->imageView[image];
            sslrImageInfo.sampler = sslrAttachment->sampler;

        VkDescriptorImageInfo ssaoImageInfo;
            ssaoImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ssaoImageInfo.imageView = ssaoAttachment->imageView[image];
            ssaoImageInfo.sampler = ssaoAttachment->sampler;

        std::vector<VkDescriptorImageInfo> blitImageInfo(postProcessing.blitAttachmentCount);
        for(uint32_t i = 0, index = 0; i < blitImageInfo.size(); i++, index++){
            blitImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            blitImageInfo[index].imageView = blitAttachments[i].imageView[image];
            blitImageInfo[index].sampler = blitAttachments[i].sampler;
        }

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &layersImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &blurImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = static_cast<uint32_t>(blitImageInfo.size());
            descriptorWrites.back().pImageInfo = blitImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &sslrImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.DescriptorSets[image];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &ssaoImageInfo;
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void postProcessingGraphics::render(uint32_t frameNumber, VkCommandBuffer commandBuffers)
{
    std::array<VkClearValue, 1> ClearValues{};
        ClearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(ClearValues.size());
        renderPassInfo.pClearValues = ClearValues.data();

    vkCmdBeginRenderPass(commandBuffers, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        postProcessingPushConst pushConst{};
            pushConst.blitFactor = postProcessing.blitFactor;
        vkCmdPushConstants(commandBuffers, postProcessing.PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(postProcessingPushConst), &pushConst);

        vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, postProcessing.Pipeline);
        vkCmdBindDescriptorSets(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, postProcessing.PipelineLayout, 0, 1, &postProcessing.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers, 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers);
}
