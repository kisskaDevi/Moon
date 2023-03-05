#include "layersCombiner.h"
#include "core/operations.h"
#include "core/texture.h"
#include "core/transformational/camera.h"

#include <array>
#include <iostream>

layersCombiner::layersCombiner()
{

}

void layersCombiner::setEmptyTexture(texture* emptyTexture)
{
    this->emptyTexture = emptyTexture;
}

void layersCombiner::setExternalPath(const std::string &path)
{
    combiner.ExternalPath = path;
}

void layersCombiner::setTransparentLayersCount(uint32_t transparentLayersCount)
{
    combiner.transparentLayersCount = transparentLayersCount;
}

void layersCombiner::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device)
{
    this->physicalDevice = physicalDevice;
    this->device = device;
}
void layersCombiner::setImageProp(imageInfo* pInfo){image = *pInfo;}
void layersCombiner::setAttachments(uint32_t attachmentsCount, attachments* pAttachments)
{
    this->attachmentsCount = attachmentsCount;
    this->pAttachments = pAttachments;
}

void layersCombiner::createAttachments(uint32_t attachmentsCount, attachments* pAttachments)
{
    for(size_t index=0; index<attachmentsCount; index++){
        pAttachments[index].create(physicalDevice,device,image.Format,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | (index==1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0),image.Extent,image.Count);
        VkSamplerCreateInfo samplerInfo{};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.anisotropyEnable = VK_TRUE;
            samplerInfo.maxAnisotropy = 1.0f;
            samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
            samplerInfo.unnormalizedCoordinates = VK_FALSE;
            samplerInfo.compareEnable = VK_FALSE;
            samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            samplerInfo.minLod = 0.0f;
            samplerInfo.maxLod = 0.0f;
            samplerInfo.mipLodBias = 0.0f;
        vkCreateSampler(*device, &samplerInfo, nullptr, &pAttachments[index].sampler);
    }
}

void layersCombiner::Combiner::Destroy(VkDevice* device)
{
    if(Pipeline)            {vkDestroyPipeline(*device, Pipeline, nullptr); Pipeline = VK_NULL_HANDLE;}
    if(PipelineLayout)      {vkDestroyPipelineLayout(*device, PipelineLayout,nullptr); PipelineLayout = VK_NULL_HANDLE;}
    if(DescriptorSetLayout) {vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr); DescriptorSetLayout = VK_NULL_HANDLE;}
    if(DescriptorPool)      {vkDestroyDescriptorPool(*device, DescriptorPool, nullptr); DescriptorPool = VK_NULL_HANDLE;}
}

void layersCombiner::destroy()
{
    combiner.Destroy(device);

    if(renderPass) {vkDestroyRenderPass(*device, renderPass, nullptr); renderPass = VK_NULL_HANDLE;}
    for(size_t i = 0; i< framebuffers.size();i++)
        if(framebuffers[i]) vkDestroyFramebuffer(*device, framebuffers[i],nullptr);
    framebuffers.resize(0);
}

void layersCombiner::createRenderPass()
{
    std::array<VkAttachmentDescription,2> attachments{};
    for(uint32_t index = 0; index < attachments.size(); index++)
        attachments[index] = attachments::imageDescription(image.Format, (index==1 ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));

    uint32_t index = 0;
    std::array<VkAttachmentReference,2> attachmentRef;
        attachmentRef[index].attachment = 0;
        attachmentRef[index].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    index++;
        attachmentRef[index].attachment = 1;
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

void layersCombiner::createFramebuffers()
{
    framebuffers.resize(image.Count);
    for(size_t i = 0; i < image.Count; i++){
        std::vector<VkImageView> attachments(attachmentsCount);
        for(size_t attachmentNumber=0; attachmentNumber<attachmentsCount; attachmentNumber++){
            attachments[attachmentNumber] = pAttachments[attachmentNumber].imageView[i];
        }
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

void layersCombiner::createPipelines()
{
    combiner.createDescriptorSetLayout(device);
    combiner.createPipeline(device,&image,&renderPass);
}

void layersCombiner::Combiner::createDescriptorSetLayout(VkDevice* device)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(VkDescriptorSetLayoutBinding{});
        bindings.back().binding = bindings.size() - 1;
        bindings.back().descriptorCount = 1;
        bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
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
        bindings.back().descriptorCount = 1;
        bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings.back().pImmutableSamplers = nullptr;
        bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings.push_back(VkDescriptorSetLayoutBinding{});
        bindings.back().binding = bindings.size() - 1;
        bindings.back().descriptorCount = transparentLayersCount;
        bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings.back().pImmutableSamplers = nullptr;
        bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings.push_back(VkDescriptorSetLayoutBinding{});
        bindings.back().binding = bindings.size() - 1;
        bindings.back().descriptorCount = transparentLayersCount;
        bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings.back().pImmutableSamplers = nullptr;
        bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings.push_back(VkDescriptorSetLayoutBinding{});
        bindings.back().binding = bindings.size() - 1;
        bindings.back().descriptorCount = transparentLayersCount;
        bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings.back().pImmutableSamplers = nullptr;
        bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings.push_back(VkDescriptorSetLayoutBinding{});
        bindings.back().binding = bindings.size() - 1;
        bindings.back().descriptorCount = transparentLayersCount;
        bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings.back().pImmutableSamplers = nullptr;
        bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings.push_back(VkDescriptorSetLayoutBinding{});
        bindings.back().binding = bindings.size() - 1;
        bindings.back().descriptorCount = transparentLayersCount;
        bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings.back().pImmutableSamplers = nullptr;
        bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings.push_back(VkDescriptorSetLayoutBinding{});
        bindings.back().binding = bindings.size() - 1;
        bindings.back().descriptorCount = 1;
        bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings.back().pImmutableSamplers = nullptr;
        bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(*device, &layoutInfo, nullptr, &DescriptorSetLayout);
}

void layersCombiner::Combiner::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    uint32_t specializationData = transparentLayersCount;
    VkSpecializationMapEntry specializationMapEntry{};
        specializationMapEntry.constantID = 0;
        specializationMapEntry.offset = 0;
        specializationMapEntry.size = sizeof(uint32_t);
    VkSpecializationInfo specializationInfo;
        specializationInfo.mapEntryCount = 1;
        specializationInfo.pMapEntries = &specializationMapEntry;
        specializationInfo.dataSize = sizeof(specializationData);
        specializationInfo.pData = &specializationData;

    uint32_t index = 0;
    auto vertShaderCode = ShaderModule::readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\layersCombiner\\layersCombinerVert.spv");
    auto fragShaderCode = ShaderModule::readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\layersCombiner\\layersCombinerFrag.spv");
    VkShaderModule vertShaderModule = ShaderModule::create(device, vertShaderCode);
    VkShaderModule fragShaderModule = ShaderModule::create(device, fragShaderCode);
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
        shaderStages[index].pSpecializationInfo = &specializationInfo;

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
        pipelineInfo.renderPass = *pRenderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.pDepthStencilState = &depthStencil;
    vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Pipeline);

    vkDestroyShaderModule(*device, fragShaderModule, nullptr);
    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
}

void layersCombiner::createDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(combiner.transparentLayersCount*image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(combiner.transparentLayersCount*image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(combiner.transparentLayersCount*image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(combiner.transparentLayersCount*image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(combiner.transparentLayersCount*image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(combiner.transparentLayersCount*image.Count);
    vkCreateDescriptorPool(*device, &poolInfo, nullptr, &combiner.DescriptorPool);
}

void layersCombiner::createDescriptorSets()
{
    combiner.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, combiner.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = combiner.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(*device, &allocInfo, combiner.DescriptorSets.data());
}

void layersCombiner::updateDescriptorSets(DeferredAttachments deferredAttachments, DeferredAttachments* transparencyLayers, attachments* skybox, camera* cameraObject)
{
    for (size_t i = 0; i < image.Count; i++)
    {
        VkDescriptorBufferInfo bufferInfo;
            bufferInfo.buffer = cameraObject->getBuffer(i);
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo colorImageInfo;
            colorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            colorImageInfo.imageView = deferredAttachments.image.imageView[i];
            colorImageInfo.sampler = deferredAttachments.image.sampler;

        VkDescriptorImageInfo bloomImageInfo;
            bloomImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            bloomImageInfo.imageView = deferredAttachments.bloom.imageView[i];
            bloomImageInfo.sampler = deferredAttachments.bloom.sampler;

        VkDescriptorImageInfo positionImageInfo;
            positionImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            positionImageInfo.imageView = deferredAttachments.GBuffer.position.imageView[i];
            positionImageInfo.sampler = deferredAttachments.GBuffer.position.sampler;

        VkDescriptorImageInfo normalImageInfo;
            normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            normalImageInfo.imageView = deferredAttachments.GBuffer.normal.imageView[i];
            normalImageInfo.sampler = deferredAttachments.GBuffer.normal.sampler;

        VkDescriptorImageInfo depthImageInfo;
            depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            depthImageInfo.imageView = deferredAttachments.depth.imageView[i];
            depthImageInfo.sampler = deferredAttachments.depth.sampler;

        VkDescriptorImageInfo skyboxImageInfo;
            skyboxImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            skyboxImageInfo.imageView = skybox->imageView.size() ? skybox->imageView[i] : *emptyTexture->getTextureImageView();
            skyboxImageInfo.sampler = skybox->sampler ? skybox->sampler : *emptyTexture->getTextureSampler();

        std::vector<VkDescriptorImageInfo> colorLayersImageInfo(combiner.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> bloomLayersImageInfo(combiner.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> positionLayersImageInfo(combiner.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> normalLayersImageInfo(combiner.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> depthLayersImageInfo(combiner.transparentLayersCount);

        for(uint32_t index = 0; index < combiner.transparentLayersCount; index++){
            colorLayersImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            colorLayersImageInfo[index].imageView = transparencyLayers[index].image.imageView[i];
            colorLayersImageInfo[index].sampler = transparencyLayers[index].image.sampler;

            bloomLayersImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            bloomLayersImageInfo[index].imageView = transparencyLayers[index].bloom.imageView[i];
            bloomLayersImageInfo[index].sampler = transparencyLayers[index].bloom.sampler;

            positionLayersImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            positionLayersImageInfo[index].imageView = transparencyLayers[index].GBuffer.position.imageView[i];
            positionLayersImageInfo[index].sampler = transparencyLayers[index].GBuffer.position.sampler;

            normalLayersImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            normalLayersImageInfo[index].imageView = transparencyLayers[index].GBuffer.normal.imageView[i];
            normalLayersImageInfo[index].sampler = transparencyLayers[index].GBuffer.normal.sampler;

            depthLayersImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            depthLayersImageInfo[index].imageView = transparencyLayers[index].depth.imageView[i];
            depthLayersImageInfo[index].sampler = transparencyLayers[index].depth.sampler;
        }

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &colorImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &bloomImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &positionImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &normalImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &depthImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = combiner.transparentLayersCount;
            descriptorWrites.back().pImageInfo = colorLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = combiner.transparentLayersCount;
            descriptorWrites.back().pImageInfo = bloomLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = combiner.transparentLayersCount;
            descriptorWrites.back().pImageInfo = positionLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = combiner.transparentLayersCount;
            descriptorWrites.back().pImageInfo = normalLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = combiner.transparentLayersCount;
            descriptorWrites.back().pImageInfo = depthLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &skyboxImageInfo;
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void layersCombiner::createCommandBuffers(VkCommandPool commandPool)
{
    commandBuffers.resize(image.Count);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(image.Count);
    vkAllocateCommandBuffers(*device, &allocInfo, commandBuffers.data());
}

void layersCombiner::beginCommandBuffer(uint32_t frameNumber){
    vkResetCommandBuffer(commandBuffers[frameNumber],0);

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

    vkBeginCommandBuffer(commandBuffers[frameNumber], &beginInfo);
}

void layersCombiner::endCommandBuffer(uint32_t frameNumber){
    vkEndCommandBuffer(commandBuffers[frameNumber]);
}

void layersCombiner::updateCommandBuffer(uint32_t frameNumber)
{
        std::array<VkClearValue, 2> ClearValues{};
        for(uint32_t index = 0; index < ClearValues.size(); index++)
            ClearValues[index].color = pAttachments->clearValue.color;

        VkRenderPassBeginInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = framebuffers[frameNumber];
            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = image.Extent;
            renderPassInfo.clearValueCount = static_cast<uint32_t>(ClearValues.size());
            renderPassInfo.pClearValues = ClearValues.data();

        vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, combiner.Pipeline);
            vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, combiner.PipelineLayout, 0, 1, &combiner.DescriptorSets[frameNumber], 0, nullptr);
            vkCmdDraw(commandBuffers[frameNumber], 6, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffers[frameNumber]);
}

VkCommandBuffer& layersCombiner::getCommandBuffer(uint32_t frameNumber)
{
    return commandBuffers[frameNumber];
}
