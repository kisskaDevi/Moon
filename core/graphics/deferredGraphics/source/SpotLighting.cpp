#include "../graphics.h"
#include "core/operations.h"
#include "core/transformational/light.h"
#include "../bufferObjects.h"

#include <array>
#include <iostream>

void deferredGraphics::SpotLighting::Destroy(VkDevice* device)
{
    vkDestroyDescriptorSetLayout(*device, LightDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(*device, DescriptorPool, nullptr);
    vkDestroyPipeline(*device, Pipeline, nullptr);
    vkDestroyPipeline(*device, ScatteringPipeline, nullptr);
    vkDestroyPipeline(*device, AmbientPipeline, nullptr);
    vkDestroyPipelineLayout(*device, PipelineLayout, nullptr);
    vkDestroyPipelineLayout(*device, AmbientPipelineLayout, nullptr);

    for (size_t i = 0; i < uniformBuffers.size(); i++)
    {
        vkDestroyBuffer(*device, uniformBuffers[i], nullptr);
        vkFreeMemory(*device, uniformBuffersMemory[i], nullptr);
    }
}

void deferredGraphics::SpotLighting::createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount)
{
    uniformBuffers.resize(imageCount);
    uniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
        createBuffer(   physicalDevice,
                        device,
                        sizeof(UniformBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        uniformBuffers[i],
                        uniformBuffersMemory[i]);
}

void deferredGraphics::SpotLighting::createDescriptorSetLayout(VkDevice* device)
{
    uint32_t index = 0;

    std::array<VkDescriptorSetLayoutBinding,6> Binding{};
    for(index = 0; index<5;index++)
    {
        Binding.at(index).binding = index;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        Binding.at(index).descriptorCount = 1;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    }
        Binding.at(index).binding = index;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        Binding.at(index).descriptorCount = 1;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(Binding.size());
        layoutInfo.pBindings = Binding.data();
    if (vkCreateDescriptorSetLayout(*device, &layoutInfo, nullptr, &DescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass descriptor set layout!");

    createSpotLightDescriptorSetLayout(device,&LightDescriptorSetLayout);
}

void deferredGraphics::SpotLighting::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\SpotLightingVert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\SpotLightingFrag.spv");
    VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);
    std::array<VkPipelineShaderStageCreateInfo,2> shaderStages{};
        shaderStages[index].pName = "main";
        shaderStages[index].module = fragShaderModule;
        shaderStages[index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    index++;
        shaderStages[index].pName = "main";
        shaderStages[index].module = vertShaderModule;
        shaderStages[index].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

    index = 0;
    std::array<VkViewport,1> viewport{};
        viewport[index].x = 0.0f;
        viewport[index].y = 0.0f;
        viewport[index].width = (float) pInfo->Extent.width;
        viewport[index].height = (float) pInfo->Extent.height;
        viewport[index].minDepth = 0.0f;
        viewport[index].maxDepth = 1.0f;
    std::array<VkRect2D,1> scissor{};
        scissor[index].offset = {0, 0};
        scissor[index].extent = pInfo->Extent;
    VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = static_cast<uint32_t>(viewport.size());
        viewportState.pViewports = viewport.data();
        viewportState.scissorCount = static_cast<uint32_t>(scissor.size());
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
        multisampling.rasterizationSamples = pInfo->Samples;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

    std::array<VkPipelineColorBlendAttachmentState,3> colorBlendAttachment;
    for(uint32_t index=0;index<colorBlendAttachment.size();index++)
    {
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_TRUE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].alphaBlendOp = VK_BLEND_OP_MAX;
    }
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

    index = 0;
    std::array<VkPushConstantRange,1> pushConstantRange{};
        pushConstantRange[index].stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange[index].offset = 0;
        pushConstantRange[index].size = sizeof(lightPassPushConst);
    std::array<VkDescriptorSetLayout,2> SetLayouts = {DescriptorSetLayout,LightDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    if (vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass pipeline layout!");

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
        pipelineInfo[index].subpass = 1;
        pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo[index].pDepthStencilState = &depthStencil;
    if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass graphics pipeline!");

    index = 0;
    auto vertShaderCodeScattering = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\SpotLightingVert.spv");
    auto fragShaderCodeScattering = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\SpotLightingScatteringFrag.spv");
    VkShaderModule vertShaderModuleScattering = createShaderModule(device, vertShaderCodeScattering);
    VkShaderModule fragShaderModuleScattering = createShaderModule(device, fragShaderCodeScattering);
    std::array<VkPipelineShaderStageCreateInfo,2> shaderStagesScattering{};
        shaderStagesScattering[index].pName = "main";
        shaderStagesScattering[index].module = fragShaderModuleScattering;
        shaderStagesScattering[index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStagesScattering[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    index++;
        shaderStagesScattering[index].pName = "main";
        shaderStagesScattering[index].module = vertShaderModuleScattering;
        shaderStagesScattering[index].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStagesScattering[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

    index = 0;
    std::array<VkGraphicsPipelineCreateInfo,1> pipelineInfoScattering{};
        pipelineInfoScattering[index].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfoScattering[index].stageCount = static_cast<uint32_t>(shaderStagesScattering.size());
        pipelineInfoScattering[index].pStages = shaderStagesScattering.data();
        pipelineInfoScattering[index].pVertexInputState = &vertexInputInfo;
        pipelineInfoScattering[index].pInputAssemblyState = &inputAssembly;
        pipelineInfoScattering[index].pViewportState = &viewportState;
        pipelineInfoScattering[index].pRasterizationState = &rasterizer;
        pipelineInfoScattering[index].pMultisampleState = &multisampling;
        pipelineInfoScattering[index].pColorBlendState = &colorBlending;
        pipelineInfoScattering[index].layout = PipelineLayout;
        pipelineInfoScattering[index].renderPass = *pRenderPass;
        pipelineInfoScattering[index].subpass = 1;
        pipelineInfoScattering[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfoScattering[index].pDepthStencilState = &depthStencil;
    if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfoScattering.size()), pipelineInfoScattering.data(), nullptr, &ScatteringPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass graphics pipeline!");

    index = 0;
    auto vertShaderCodeAmbient = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\AmbientSpotLightingVert.spv");
    auto fragShaderCodeAmbient = readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\SpotLightingPass\\AmbientSpotLightingFrag.spv");
    VkShaderModule vertShaderModuleAmbient = createShaderModule(device, vertShaderCodeAmbient);
    VkShaderModule fragShaderModuleAmbient = createShaderModule(device, fragShaderCodeAmbient);
    std::array<VkPipelineShaderStageCreateInfo,2> shaderStagesAmbient{};
        shaderStagesAmbient[index].pName = "main";
        shaderStagesAmbient[index].module = fragShaderModuleAmbient;
        shaderStagesAmbient[index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStagesAmbient[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    index++;
        shaderStagesAmbient[index].pName = "main";
        shaderStagesAmbient[index].module = vertShaderModuleAmbient;
        shaderStagesAmbient[index].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStagesAmbient[index].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

    VkPipelineDepthStencilStateCreateInfo AmbientDepthStencil{};
        AmbientDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        AmbientDepthStencil.depthTestEnable = VK_FALSE;
        AmbientDepthStencil.depthWriteEnable = VK_FALSE;
        AmbientDepthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        AmbientDepthStencil.depthBoundsTestEnable = VK_FALSE;
        AmbientDepthStencil.minDepthBounds = 0.0f;
        AmbientDepthStencil.maxDepthBounds = 1.0f;
        AmbientDepthStencil.stencilTestEnable = VK_FALSE;
        AmbientDepthStencil.front = {};
        AmbientDepthStencil.back = {};

    index = 0;
    std::array<VkPushConstantRange,1> ambientPushConstantRange{};
        ambientPushConstantRange[index].stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        ambientPushConstantRange[index].offset = 0;
        ambientPushConstantRange[index].size = sizeof(lightPassPushConst);
    std::array<VkDescriptorSetLayout,1> ambientSetLayouts = {DescriptorSetLayout};
    VkPipelineLayoutCreateInfo ambientPipelineLayoutInfo{};
        ambientPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ambientPipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(ambientSetLayouts.size());
        ambientPipelineLayoutInfo.pSetLayouts = ambientSetLayouts.data();
        ambientPipelineLayoutInfo.pushConstantRangeCount = 1;
        ambientPipelineLayoutInfo.pPushConstantRanges = ambientPushConstantRange.data();
    if (vkCreatePipelineLayout(*device, &ambientPipelineLayoutInfo, nullptr, &AmbientPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass pipeline layout!");

    index = 0;
    std::array<VkGraphicsPipelineCreateInfo,1> pipelineInfoAmbient{};
        pipelineInfoAmbient[index].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfoAmbient[index].stageCount = static_cast<uint32_t>(shaderStagesAmbient.size());
        pipelineInfoAmbient[index].pStages = shaderStagesAmbient.data();
        pipelineInfoAmbient[index].pVertexInputState = &vertexInputInfo;
        pipelineInfoAmbient[index].pInputAssemblyState = &inputAssembly;
        pipelineInfoAmbient[index].pViewportState = &viewportState;
        pipelineInfoAmbient[index].pRasterizationState = &rasterizer;
        pipelineInfoAmbient[index].pMultisampleState = &multisampling;
        pipelineInfoAmbient[index].pColorBlendState = &colorBlending;
        pipelineInfoAmbient[index].layout = AmbientPipelineLayout;
        pipelineInfoAmbient[index].renderPass = *pRenderPass;
        pipelineInfoAmbient[index].subpass = 1;
        pipelineInfoAmbient[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfoAmbient[index].pDepthStencilState = &AmbientDepthStencil;
    if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfoAmbient.size()), pipelineInfoAmbient.data(), nullptr, &AmbientPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass graphics pipeline!");

    vkDestroyShaderModule(*device, fragShaderModuleAmbient, nullptr);
    vkDestroyShaderModule(*device, vertShaderModuleAmbient, nullptr);
    vkDestroyShaderModule(*device, fragShaderModule, nullptr);
    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
    vkDestroyShaderModule(*device, fragShaderModuleScattering, nullptr);
    vkDestroyShaderModule(*device, vertShaderModuleScattering, nullptr);
}

void deferredGraphics::createSpotLightingDescriptorPool()
{
    uint32_t index = 0;

    std::array<VkDescriptorPoolSize,6> poolSizes{};
        for(uint32_t i = 0;i<5;i++,index++)
            poolSizes[index] = {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, static_cast<uint32_t>(image.Count)};
        poolSizes[index] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(image.Count)};
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    if (vkCreateDescriptorPool(*device, &poolInfo, nullptr, &spotLighting.DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass descriptor pool!");
}

void deferredGraphics::createSpotLightingDescriptorSets()
{
    spotLighting.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, spotLighting.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = spotLighting.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(*device, &allocInfo, spotLighting.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate SpotLightingPass descriptor sets!");
}

void deferredGraphics::updateSpotLightingDescriptorSets()
{
    for (size_t i = 0; i < image.Count; i++)
    {
        uint32_t index = 0;

        std::array<VkDescriptorImageInfo,5> imageInfo{};
        for(index = 0; index<4;index++)
        {
            imageInfo.at(index).imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.at(index).imageView = Attachments.at(3+index).imageView.at(i);
            imageInfo.at(index).sampler = VK_NULL_HANDLE;
        }
        imageInfo.at(index).imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.at(index).imageView = this->depthAttachment.imageView;
        imageInfo.at(index).sampler = VK_NULL_HANDLE;
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = spotLighting.uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

        std::array<VkWriteDescriptorSet,6> descriptorWrites{};
        for(index = 0; index<5;index++)
        {
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = spotLighting.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pImageInfo = &imageInfo.at(index);
        }
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = spotLighting.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void deferredGraphics::SpotLighting::render(uint32_t frameNumber, VkCommandBuffer commandBuffers)
{
    for(uint32_t lightNumber = 0; lightNumber<lightSources.size();lightNumber++)
    {
        if(lightSources[lightNumber]->isScatteringEnable()&&enableScattering)
            vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, ScatteringPipeline);
        else
            vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);

        lightPassPushConst pushConst{};
            pushConst.minAmbientFactor = minAmbientFactor;
        vkCmdPushConstants(commandBuffers, PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(lightPassPushConst), &pushConst);

        std::vector<VkDescriptorSet> descriptorSets = {DescriptorSets[frameNumber],lightSources[lightNumber]->getDescriptorSets()[frameNumber]};
        vkCmdBindDescriptorSets(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);

        vkCmdDraw(commandBuffers, 18, 1, 0, 0);
    }

    lightPassPushConst pushConst{};
        pushConst.minAmbientFactor = minAmbientFactor;
    vkCmdPushConstants(commandBuffers, PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(lightPassPushConst), &pushConst);

    vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, AmbientPipeline);
    vkCmdBindDescriptorSets(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, AmbientPipelineLayout, 0, 1, &DescriptorSets[frameNumber], 0, nullptr);
    vkCmdDraw(commandBuffers, 6, 1, 0, 0);
}

void deferredGraphics::updateSpotLightUbo(uint32_t imageIndex)
{
    for(auto lightSource: spotLighting.lightSources)
        lightSource->updateLightBuffer(device,imageIndex);
}

void deferredGraphics::updateSpotLightCmd(uint32_t imageIndex)
{
    std::vector<object*> objects(base.objects.size()+oneColor.objects.size()+stencil.objects.size());

    uint32_t counter = 0;
    for(auto object: base.objects){
        objects[counter] = object;
        counter++;
    }
    for(auto object: oneColor.objects){
        objects[counter] = object;
        counter++;
    }
    for(auto object: stencil.objects){
        objects[counter] = object;
        counter++;
    }

    for(auto lightSource: spotLighting.lightSources)
        if(lightSource->isShadowEnable())
            lightSource->updateShadowCommandBuffer(imageIndex,objects);
}

void deferredGraphics::getSpotLightCommandbuffers(std::vector<VkCommandBuffer>* commandbufferSet, uint32_t imageIndex)
{
    for(auto lightSource: spotLighting.lightSources)
        if(lightSource->isShadowEnable())
            commandbufferSet->push_back(lightSource->getShadowCommandBuffer()[imageIndex]);
}
