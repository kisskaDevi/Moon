#include "core/graphics/graphics.h"
#include "core/operations.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/light.h"
#include "core/graphics/shadowGraphics.h"

#include <array>

void graphics::Second::Destroy(VkApplication *app)
{
    vkDestroyDescriptorSetLayout(app->getDevice(), DescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(app->getDevice(), DescriptorPool, nullptr);
    vkDestroyPipeline(app->getDevice(), Pipeline, nullptr);
    vkDestroyPipeline(app->getDevice(), ScatteringPipeline, nullptr);
    vkDestroyPipeline(app->getDevice(), AmbientPipeline, nullptr);
    vkDestroyPipelineLayout(app->getDevice(), PipelineLayout, nullptr);

    for (size_t i = 0; i < uniformBuffers.size(); i++)
    {
        vkDestroyBuffer(app->getDevice(), lightUniformBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), lightUniformBuffersMemory[i], nullptr);

        vkDestroyBuffer(app->getDevice(), uniformBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), uniformBuffersMemory[i], nullptr);

        vkDestroyBuffer(app->getDevice(), nodeMaterialUniformBuffers[i], nullptr);
        vkFreeMemory(app->getDevice(), nodeMaterialUniformBuffersMemory[i], nullptr);
    }
}

void graphics::Second::createDescriptorSetLayout(VkApplication *app)
{
    uint32_t index = 0;

    std::array<VkDescriptorSetLayoutBinding,8> Binding{};
    for(index = 0; index<4;index++)
    {
        Binding.at(index).binding = index;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        Binding.at(index).descriptorCount = 1;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    }
        Binding.at(index).binding = index;
        Binding.at(index).descriptorCount = 1;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    index++;
        Binding.at(index).binding = index;
        Binding.at(index).descriptorCount = MAX_LIGHT_SOURCE_COUNT;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    index++;
        Binding.at(index).binding = index;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        Binding.at(index).descriptorCount = 1;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    index++;
        Binding.at(index).binding = index;
        Binding.at(index).descriptorCount = MAX_LIGHT_SOURCE_COUNT;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(Binding.size());
        layoutInfo.pBindings = Binding.data();
    if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &DescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create SecondPass descriptor set layout!");
}

void graphics::Second::createPipeline(VkApplication *app, graphicsInfo info)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\secondPass\\secondvert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\secondPass\\secondfrag.spv");
    VkShaderModule vertShaderModule = createShaderModule(app, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(app, fragShaderCode);
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
        viewport[index].width = (float) info.extent.width;
        viewport[index].height = (float) info.extent.height;
        viewport[index].minDepth = 0.0f;
        viewport[index].maxDepth = 1.0f;
    std::array<VkRect2D,1> scissor{};
        scissor[index].offset = {0, 0};
        scissor[index].extent = info.extent;
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
        multisampling.rasterizationSamples = info.msaaSamples;
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

    index=0;
    std::array<VkPushConstantRange,1> pushConstantRange{};
        pushConstantRange[index].stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange[index].offset = 0;
        pushConstantRange[index].size = sizeof(secondPassPushConst);
    std::array<VkDescriptorSetLayout,1> SetLayouts = {DescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create SecondPass pipeline layout!");

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
        pipelineInfo[index].renderPass = info.renderPass;
        pipelineInfo[index].subpass = 1;
        pipelineInfo[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo[index].pDepthStencilState = &depthStencil;
    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create SecondPass graphics pipeline!");

    auto vertShaderCodeScattering = readFile(ExternalPath + "core\\graphics\\shaders\\secondPass\\secondvert.spv");
    auto fragShaderCodeScattering = readFile(ExternalPath + "core\\graphics\\shaders\\secondPass\\secondScatteringfrag.spv");
    VkShaderModule vertShaderModuleScattering = createShaderModule(app, vertShaderCodeScattering);
    VkShaderModule fragShaderModuleScattering = createShaderModule(app, fragShaderCodeScattering);
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
        pipelineInfoScattering[index].renderPass = info.renderPass;
        pipelineInfoScattering[index].subpass = 1;
        pipelineInfoScattering[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfoScattering[index].pDepthStencilState = &depthStencil;
    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfoScattering.size()), pipelineInfoScattering.data(), nullptr, &ScatteringPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create SecondPass graphics pipeline!");

    auto vertShaderCodeAmbient = readFile(ExternalPath + "core\\graphics\\shaders\\secondPass\\AmbientSecondvert.spv");
    auto fragShaderCodeAmbient = readFile(ExternalPath + "core\\graphics\\shaders\\secondPass\\Ambientsecondfrag.spv");
    VkShaderModule vertShaderModuleAmbient = createShaderModule(app, vertShaderCodeAmbient);
    VkShaderModule fragShaderModuleAmbient = createShaderModule(app, fragShaderCodeAmbient);
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
        pipelineInfoAmbient[index].layout = PipelineLayout;
        pipelineInfoAmbient[index].renderPass = info.renderPass;
        pipelineInfoAmbient[index].subpass = 1;
        pipelineInfoAmbient[index].basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfoAmbient[index].pDepthStencilState = &AmbientDepthStencil;
    if (vkCreateGraphicsPipelines(app->getDevice(), VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfoAmbient.size()), pipelineInfoAmbient.data(), nullptr, &AmbientPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create SecondPass graphics pipeline!");

    vkDestroyShaderModule(app->getDevice(), fragShaderModuleAmbient, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModuleAmbient, nullptr);
    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), fragShaderModuleScattering, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModuleScattering, nullptr);
}

void graphics::createSecondDescriptorPool()
{
    uint32_t index = 0;

    std::array<VkDescriptorPoolSize,6+2*MAX_LIGHT_SOURCE_COUNT> poolSizes{};

        for(uint32_t i = 0;i<4;i++,index++)
            poolSizes[index] = {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, static_cast<uint32_t>(image.Count)};

        poolSizes[index] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(image.Count)};
    index++;
        poolSizes[index] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(image.Count)};
    index++;

        for(size_t i=0;i<MAX_LIGHT_SOURCE_COUNT;i++,index++)
            poolSizes[index] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(image.Count)};

        for(size_t i=0;i<MAX_LIGHT_SOURCE_COUNT;i++,index++)
            poolSizes[index] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(image.Count)};

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &second.DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create SecondPass descriptor pool!");
}

void graphics::createSecondDescriptorSets()
{
    second.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, second.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = second.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, second.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate SecondPass descriptor sets!");
}

void graphics::updateSecondDescriptorSets(const std::vector<light<spotLight>*> & lightSources)
{
    for (size_t i = 0; i < image.Count; i++)
    {
        uint32_t index = 0;

        std::array<VkDescriptorImageInfo,4> imageInfo{};
        for(index = 0; index<4;index++)
        {
            imageInfo.at(index).imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.at(index).imageView = Attachments.at(3+index).imageView.at(i);
            imageInfo.at(index).sampler = VK_NULL_HANDLE;
        }
        VkDescriptorBufferInfo lightBufferInfo{};
            lightBufferInfo.buffer = second.lightUniformBuffers[i];
            lightBufferInfo.offset = 0;
            lightBufferInfo.range = sizeof(LightUniformBufferObject);
        std::array<VkDescriptorImageInfo,MAX_LIGHT_SOURCE_COUNT> shadowImageInfo{};
            for (size_t j = 0; j < lightSources.size(); j++)
            {
                shadowImageInfo[j].imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                shadowImageInfo[j].imageView    = lightSources.at(j)->getShadowEnable() ? lightSources.at(j)->getShadow()->getImageView() : emptyTexture->getTextureImageView();
                shadowImageInfo[j].sampler      = lightSources.at(j)->getShadowEnable() ? lightSources.at(j)->getShadow()->getSampler() : emptyTexture->getTextureSampler();
            }
            for (size_t j = lightSources.size(); j < MAX_LIGHT_SOURCE_COUNT; j++)
            {
                shadowImageInfo[j].imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                shadowImageInfo[j].imageView    = emptyTexture->getTextureImageView();
                shadowImageInfo[j].sampler      = emptyTexture->getTextureSampler();
            }
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = second.uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);
        std::array<VkDescriptorImageInfo,MAX_LIGHT_SOURCE_COUNT> lightTexture{};
            for (size_t j = 0; j < lightSources.size(); j++)
            {
                lightTexture[j].imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                lightTexture[j].imageView    = lightSources.at(j)->getTexture() ? lightSources.at(j)->getTexture()->getTextureImageView() : emptyTexture->getTextureImageView();
                lightTexture[j].sampler      = lightSources.at(j)->getTexture() ? lightSources.at(j)->getTexture()->getTextureSampler() : emptyTexture->getTextureSampler();
            }
            for (size_t j = lightSources.size(); j < MAX_LIGHT_SOURCE_COUNT; j++)
            {
                lightTexture[j].imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                lightTexture[j].imageView    = emptyTexture->getTextureImageView();
                lightTexture[j].sampler      = emptyTexture->getTextureSampler();
            }

        std::array<VkWriteDescriptorSet,8> descriptorWrites{};
        for(index = 0; index<4;index++)
        {
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = second.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pImageInfo = &imageInfo.at(index);
        }
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = second.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pBufferInfo = &lightBufferInfo;
        index++;
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = second.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.at(index).descriptorCount = static_cast<uint32_t>(shadowImageInfo.size());
            descriptorWrites.at(index).pImageInfo = shadowImageInfo.data();
        index++;
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = second.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pBufferInfo = &bufferInfo;
        index++;
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = second.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.at(index).descriptorCount = static_cast<uint32_t>(lightTexture.size());
            descriptorWrites.at(index).pImageInfo = lightTexture.data();
        vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    for(size_t i=0;i<lightSources.size();i++)
        if(lightSources[i]->getShadowEnable())
            lightSources[i]->getShadow()->updateDescriptorSets(second.lightUniformBuffers);
}

void graphics::Second::createUniformBuffers(VkApplication *app, uint32_t imageCount)
{
    lightUniformBuffers.resize(imageCount);
    lightUniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
        createBuffer(app, sizeof(LightUniformBufferObject),
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     lightUniformBuffers[i], lightUniformBuffersMemory[i]);

    uniformBuffers.resize(imageCount);
    uniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
        createBuffer(app, sizeof(UniformBufferObject),
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     uniformBuffers[i], uniformBuffersMemory[i]);

    nodeMaterialUniformBuffers.resize(imageCount);
    nodeMaterialUniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
        createBuffer(app, nodeMaterialCount*sizeof(MaterialBlock),
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     nodeMaterialUniformBuffers[i], nodeMaterialUniformBuffersMemory[i]);
}

void graphics::Second::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i, std::vector<light<spotLight> *> lightSource)
{
    for(uint32_t lightNumber = 0; lightNumber<lightSource.size();lightNumber++)
    {
        if(lightSource[lightNumber]->getScatteringEnable())
            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, ScatteringPipeline);
        else
            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);

        vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &DescriptorSets[i], 0, nullptr);
        secondPassPushConst pushConst{};
            pushConst.lightNumber = lightNumber;
        vkCmdPushConstants(commandBuffers[i], PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(secondPassPushConst), &pushConst);

        vkCmdDraw(commandBuffers[i], 18, 1, 0, 0);
    }

    vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, AmbientPipeline);
    vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &DescriptorSets[i], 0, nullptr);
    secondPassPushConst pushConst{};
    vkCmdPushConstants(commandBuffers[i], PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(secondPassPushConst), &pushConst);
    vkCmdDraw(commandBuffers[i], 6, 1, 0, 0);
}
