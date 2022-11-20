#include "blur.h"

#include "core/operations.h"

#include <array>
#include <iostream>

gaussianBlur::gaussianBlur()
{

}

void gaussianBlur::setExternalPath(const std::string &path)
{
    xblur.ExternalPath = path;
    yblur.ExternalPath = path;
}

void gaussianBlur::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool)
{
    this->physicalDevice = physicalDevice;
    this->device = device;
    this->graphicsQueue = graphicsQueue;
    this->commandPool = commandPool;
}
void gaussianBlur::setImageProp(imageInfo* pInfo)
{
    this->image = *pInfo;
}
void gaussianBlur::setAttachments(uint32_t attachmentsCount, attachments* pAttachments)
{
    this->attachmentsCount = attachmentsCount;
    this->pAttachments = pAttachments;
}

void gaussianBlur::createBufferAttachments()
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
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |VK_IMAGE_USAGE_SAMPLED_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                bufferAttachment.image[imageNumber],
                                bufferAttachment.imageMemory[imageNumber]);

        bufferAttachment.imageView[imageNumber] =
        createImageView(        device,
                                bufferAttachment.image[imageNumber],
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
    vkCreateSampler(*device, &SamplerInfo, nullptr, &bufferAttachment.sampler);
}

void gaussianBlur::createAttachments(uint32_t attachmentsCount, attachments* pAttachments)
{
    static_cast<void>(attachmentsCount);
    pAttachments->resize(image.Count);
    for(size_t Image=0; Image<image.Count; Image++)
    {
        createImage(        physicalDevice,
                            device,
                            image.Extent.width,
                            image.Extent.height,
                            1,
                            VK_SAMPLE_COUNT_1_BIT,
                            image.Format,
                            VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            pAttachments->image[Image],
                            pAttachments->imageMemory[Image]);

        createImageView(    device,
                            pAttachments->image[Image],
                            image.Format,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            1,
                            &pAttachments->imageView[Image]);
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
    vkCreateSampler(*device, &SamplerInfo, nullptr, &pAttachments->sampler);
}

void gaussianBlur::xBlur::Destroy(VkDevice* device)
{
    if(Pipeline)            vkDestroyPipeline(*device, Pipeline, nullptr);
    if(PipelineLayout)      vkDestroyPipelineLayout(*device, PipelineLayout,nullptr);
    if(DescriptorSetLayout) vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr);
    if(DescriptorPool)      vkDestroyDescriptorPool(*device, DescriptorPool, nullptr);
}

void gaussianBlur::yBlur::Destroy(VkDevice* device)
{
    if(Pipeline)            vkDestroyPipeline(*device, Pipeline, nullptr);
    if(PipelineLayout)      vkDestroyPipelineLayout(*device, PipelineLayout,nullptr);
    if(DescriptorSetLayout) vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr);
    if(DescriptorPool)      vkDestroyDescriptorPool(*device, DescriptorPool, nullptr);
}

void gaussianBlur::destroy()
{
    xblur.Destroy(device);
    yblur.Destroy(device);

    if(renderPass) vkDestroyRenderPass(*device, renderPass, nullptr);
    for(size_t i = 0; i< framebuffers.size();i++)
        if(framebuffers[i]) vkDestroyFramebuffer(*device, framebuffers[i],nullptr);

    bufferAttachment.deleteAttachment(device);
    bufferAttachment.deleteSampler(device);
}

void gaussianBlur::createRenderPass()
{
    uint32_t index = 0;

    std::array<VkAttachmentDescription,2> attachments{};
        attachments[index].format = image.Format;
        attachments[index].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[index].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[index].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[index].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[index].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[index].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[index].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    index++;
        attachments[index].format = image.Format;
        attachments[index].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[index].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[index].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[index].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[index].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[index].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[index].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    index = 0;
    std::array<VkAttachmentReference,2> firstAttachmentRef;
        firstAttachmentRef[index].attachment = 0;
        firstAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    index++;
        firstAttachmentRef[index].attachment = 1;
        firstAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    index = 0;
    std::array<VkAttachmentReference,1> secondAttachmentRef;
        secondAttachmentRef[index].attachment = 0;
        secondAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    index = 0;
    std::array<VkAttachmentReference,1> inSecondAttachmentRef;
        inSecondAttachmentRef[index].attachment = 1;
        inSecondAttachmentRef[index].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    index = 0;
    std::array<VkAttachmentReference,1> thirdAttachmentRef;
        thirdAttachmentRef[index].attachment = 0;
        thirdAttachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    index = 0;
    std::array<VkSubpassDescription,3> subpass{};
        subpass[index].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass[index].colorAttachmentCount = static_cast<uint32_t>(firstAttachmentRef.size());
        subpass[index].pColorAttachments = firstAttachmentRef.data();
    index++;
        subpass[index].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass[index].colorAttachmentCount = static_cast<uint32_t>(secondAttachmentRef.size());
        subpass[index].pColorAttachments = secondAttachmentRef.data();
        subpass[index].inputAttachmentCount = static_cast<uint32_t>(inSecondAttachmentRef.size());
        subpass[index].pInputAttachments = inSecondAttachmentRef.data();
    index++;
        subpass[index].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass[index].colorAttachmentCount = static_cast<uint32_t>(thirdAttachmentRef.size());
        subpass[index].pColorAttachments = thirdAttachmentRef.data();

    index = 0;
    std::array<VkSubpassDependency,3> dependency{};
        dependency[index].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency[index].dstSubpass = 0;
        dependency[index].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency[index].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependency[index].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency[index].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    index++;
        dependency[index].srcSubpass = 0;
        dependency[index].dstSubpass = 1;
        dependency[index].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency[index].srcAccessMask = 0;
        dependency[index].dstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency[index].dstAccessMask = 0;
    index++;
        dependency[index].srcSubpass = 1;
        dependency[index].dstSubpass = 2;
        dependency[index].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency[index].srcAccessMask = 0;
        dependency[index].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency[index].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

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

void gaussianBlur::createFramebuffers()
{
    framebuffers.resize(image.Count);
    for (size_t i = 0; i < framebuffers.size(); i++)
    {
        std::array<VkImageView,2> attachments;
            attachments[0] = pAttachments->imageView[i];
            attachments[1] = bufferAttachment.imageView[i];
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &framebuffers[i]);
    }
}

void gaussianBlur::xBlur::createDescriptorSetLayout(VkDevice* device)
{
    uint32_t index = 0;
    std::array<VkDescriptorSetLayoutBinding,1> bindings;
        bindings[index].binding = 0;
        bindings[index].descriptorCount = 1;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[index].pImmutableSamplers = nullptr;
        bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(*device, &layoutInfo, nullptr, &DescriptorSetLayout);
}

void gaussianBlur::yBlur::createDescriptorSetLayout(VkDevice* device)
{
    uint32_t index = 0;
    std::array<VkDescriptorSetLayoutBinding,1> bindings;
        bindings[index].binding = 0;
        bindings[index].descriptorCount = 1;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[index].pImmutableSamplers = nullptr;
        bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(*device, &layoutInfo, nullptr, &DescriptorSetLayout);
}

void gaussianBlur::createPipelines()
{
    xblur.createDescriptorSetLayout(device);
    xblur.createPipeline(device,&image,&renderPass);
    yblur.createDescriptorSetLayout(device);
    yblur.createPipeline(device,&image,&renderPass);
}

void gaussianBlur::xBlur::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\gaussianBlur\\xBlurVert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\gaussianBlur\\xBlurFrag.spv");
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
    std::array<VkPipelineColorBlendAttachmentState,2> colorBlendAttachment;
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_FALSE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].alphaBlendOp = VK_BLEND_OP_MAX;
    index++;
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

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &DescriptorSetLayout;
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

void gaussianBlur::yBlur::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\gaussianBlur\\yBlurVert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\gaussianBlur\\yBlurFrag.spv");
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

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &DescriptorSetLayout;
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
        pipelineInfo[index].subpass = 2;
        pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo[index].pDepthStencilState = &depthStencil;
    vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline);

    vkDestroyShaderModule(*device, fragShaderModule, nullptr);
    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
}

void gaussianBlur::createDescriptorPool()
{
    size_t index = 0;
    std::array<VkDescriptorPoolSize,1> xPoolSizes;
        xPoolSizes[index].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        xPoolSizes[index].descriptorCount = static_cast<uint32_t>(image.Count);
    VkDescriptorPoolCreateInfo xPoolInfo{};
        xPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        xPoolInfo.poolSizeCount = static_cast<uint32_t>(xPoolSizes.size());
        xPoolInfo.pPoolSizes = xPoolSizes.data();
        xPoolInfo.maxSets = static_cast<uint32_t>(image.Count);
    vkCreateDescriptorPool(*device, &xPoolInfo, nullptr, &xblur.DescriptorPool);

    index = 0;
    std::array<VkDescriptorPoolSize,1> yPoolSizes;
        yPoolSizes[index].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        yPoolSizes[index].descriptorCount = static_cast<uint32_t>(image.Count);
    VkDescriptorPoolCreateInfo yPoolInfo{};
        yPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        yPoolInfo.poolSizeCount = static_cast<uint32_t>(yPoolSizes.size());
        yPoolInfo.pPoolSizes = yPoolSizes.data();
        yPoolInfo.maxSets = static_cast<uint32_t>(image.Count);
    vkCreateDescriptorPool(*device, &yPoolInfo, nullptr, &yblur.DescriptorPool);
}

void gaussianBlur::createDescriptorSets()
{
    xblur.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> xLayouts(image.Count, xblur.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo xAllocInfo{};
        xAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        xAllocInfo.descriptorPool = xblur.DescriptorPool;
        xAllocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        xAllocInfo.pSetLayouts = xLayouts.data();
    vkAllocateDescriptorSets(*device, &xAllocInfo, xblur.DescriptorSets.data());

    yblur.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> yLayouts(image.Count, yblur.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo yAllocInfo{};
        yAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        yAllocInfo.descriptorPool = yblur.DescriptorPool;
        yAllocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        yAllocInfo.pSetLayouts = yLayouts.data();
    vkAllocateDescriptorSets(*device, &yAllocInfo, yblur.DescriptorSets.data());
}

void gaussianBlur::updateDescriptorSets(attachments* blurAttachment)
{
    for (size_t i = 0; i < image.Count; i++)
    {
        uint32_t index = 0;
        std::array<VkDescriptorImageInfo, 1> imageInfo;
            imageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[index].imageView = blurAttachment->imageView[i];
            imageInfo[index].sampler = blurAttachment->sampler;

        index = 0;
        std::array<VkWriteDescriptorSet, 1> descriptorWrites{};
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = xblur.DescriptorSets[i];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = static_cast<uint32_t>(imageInfo.size());
            descriptorWrites[index].pImageInfo = imageInfo.data();
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

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
            descriptorWrites[index].dstSet = yblur.DescriptorSets[i];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = static_cast<uint32_t>(imageInfo.size());
            descriptorWrites[index].pImageInfo = imageInfo.data();
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void gaussianBlur::render(uint32_t frameNumber, VkCommandBuffer commandBuffer)
{
    std::array<VkClearValue, 2> ClearValues{};
        ClearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        ClearValues[1].color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(ClearValues.size());
        renderPassInfo.pClearValues = ClearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, xblur.Pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, xblur.PipelineLayout, 0, 1, &xblur.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffer, 6, 1, 0, 0);

    vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, yblur.Pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, yblur.PipelineLayout, 0, 1, &yblur.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffer, 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);
}
