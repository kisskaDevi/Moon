#include "graphics.h"
#include "core/operations.h"
#include "core/transformational/gltfmodel.h"
#include "core/transformational/light.h"
#include "core/graphics/shadowGraphics.h"

void graphics::Second::Destroy(VkApplication *app)
{
    vkDestroyDescriptorSetLayout(app->getDevice(), DescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(app->getDevice(), DescriptorPool, nullptr);
    vkDestroyPipeline(app->getDevice(), Pipeline, nullptr);
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

    std::array<VkDescriptorSetLayoutBinding,10> Binding{};
    for(index = 0; index<6;index++)
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
        Binding.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
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
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        Binding.at(index).descriptorCount = 1;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(Binding.size());
        layoutInfo.pBindings = Binding.data();
    if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &DescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor set layout!");
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

    std::array<VkPipelineColorBlendAttachmentState,2> colorBlendAttachment;
    for(uint32_t index=0;index<colorBlendAttachment.size();index++)
    {
        colorBlendAttachment[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment[index].blendEnable = VK_FALSE;
        colorBlendAttachment[index].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment[index].colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
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
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};

    std::array<VkDescriptorSetLayout,1> SetLayouts = {DescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
    if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout!");

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
        throw std::runtime_error("failed to create graphics pipeline!");

    vkDestroyShaderModule(app->getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(app->getDevice(), vertShaderModule, nullptr);
}

void graphics::createSecondDescriptorPool()
{
    uint32_t index = 0;

    std::array<VkDescriptorPoolSize,9+MAX_LIGHT_SOURCE_COUNT> poolSizes{};
        for(index = 0;index<6; index++)
            poolSizes[index] = {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, static_cast<uint32_t>(imageCount)};
        for(size_t i=index;i<index+MAX_LIGHT_SOURCE_COUNT;i++)
            poolSizes[i] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(imageCount)};
    index+=MAX_LIGHT_SOURCE_COUNT;
        poolSizes[index] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(imageCount)};
    index++;
        poolSizes[index] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(imageCount)};
    index++;
        poolSizes[index] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(imageCount)};
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(imageCount);
    if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &second.DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor pool!");
}

void graphics::createSecondDescriptorSets(const std::vector<light<spotLight>*> & lightSource)
{
    second.DescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> layouts(imageCount, second.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = second.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, second.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate descriptor sets!");

    for (size_t i = 0; i < imageCount; i++)
    {
        uint32_t index = 0;

        std::array<VkDescriptorImageInfo,6> imageInfo{};
        for(index = 0; index<6;index++)
        {
            imageInfo.at(index).imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.at(index).imageView = Attachments.at(2+index).imageView.at(i);
            imageInfo.at(index).sampler = VK_NULL_HANDLE;
        }
        VkDescriptorBufferInfo lightBufferInfo{};
            lightBufferInfo.buffer = second.lightUniformBuffers[i];
            lightBufferInfo.offset = 0;
            lightBufferInfo.range = sizeof(LightUniformBufferObject);
        std::array<VkDescriptorImageInfo,MAX_LIGHT_SOURCE_COUNT> shadowImageInfo{};
            for (size_t j = 0; j < lightSource.size(); j++)
            {
                shadowImageInfo[j].imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                shadowImageInfo[j].imageView    = lightSource.at(j)->getShadowEnable() ? lightSource.at(j)->getShadow()->getImageView() : emptyTexture->getTextureImageView();
                shadowImageInfo[j].sampler      = lightSource.at(j)->getShadowEnable() ? lightSource.at(j)->getShadow()->getSampler() : emptyTexture->getTextureSampler();
            }
            for (size_t j = lightSource.size(); j < MAX_LIGHT_SOURCE_COUNT; j++)
            {
                shadowImageInfo[j].imageLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                shadowImageInfo[j].imageView    = emptyTexture->getTextureImageView();
                shadowImageInfo[j].sampler      = emptyTexture->getTextureSampler();
            }
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = second.uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(SecondUniformBufferObject);
        VkDescriptorBufferInfo MaterialBufferInfo{};
            MaterialBufferInfo.buffer = second.nodeMaterialUniformBuffers[i];
            MaterialBufferInfo.offset = 0;
            MaterialBufferInfo.range = second.nodeMaterialCount * sizeof(PushConstBlockMaterial);

        std::array<VkWriteDescriptorSet,10> descriptorWrites{};
        for(index = 0; index<6;index++)
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
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pBufferInfo = &MaterialBufferInfo;
        vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    for(size_t i=0;i<lightSource.size();i++)
        if(lightSource[i]->getShadowEnable())
            lightSource[i]->getShadow()->createDescriptorSets(second.lightUniformBuffers);
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
        createBuffer(app, sizeof(SecondUniformBufferObject),
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     uniformBuffers[i], uniformBuffersMemory[i]);

    nodeMaterialUniformBuffers.resize(imageCount);
    nodeMaterialUniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
        createBuffer(app, nodeMaterialCount*sizeof(PushConstBlockMaterial),
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     nodeMaterialUniformBuffers[i], nodeMaterialUniformBuffersMemory[i]);
}

void graphics::Second::render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i)
{
    vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &DescriptorSets[i], 0, nullptr);
    vkCmdDraw(commandBuffers[i], 6, 1, 0, 0);
}
