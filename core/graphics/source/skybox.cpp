#include "core/graphics/graphics.h"
#include "core/operations.h"
#include "core/transformational/object.h"
#include "core/transformational/gltfmodel.h"

#include <array>

void deferredGraphics::Skybox::Destroy(VkDevice* device)
{
    vkDestroyPipeline(*device, Pipeline, nullptr);
    vkDestroyPipelineLayout(*device, PipelineLayout,nullptr);
    vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(*device, DescriptorPool, nullptr);

    for (size_t i = 0; i < uniformBuffers.size(); i++)
    {
        vkDestroyBuffer(*device, uniformBuffers[i], nullptr);
        vkFreeMemory(*device, uniformBuffersMemory[i], nullptr);
    }
}

void deferredGraphics::Skybox::createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount)
{
    uniformBuffers.resize(imageCount);
    uniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++){
        createBuffer(   physicalDevice,
                        device,
                        sizeof(SkyboxUniformBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        uniformBuffers[i],
                        uniformBuffersMemory[i]);
    }
}

void deferredGraphics::Skybox::createDescriptorSetLayout(VkDevice* device)
{
    uint32_t index = 0;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[index].binding = 0;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        bindings[index].pImmutableSamplers = nullptr;
    index++;
        bindings[index].binding = 1;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[index].pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(*device, &layoutInfo, nullptr, &DescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create Skybox descriptor set layout!");
}

void deferredGraphics::Skybox::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    uint32_t index = 0;

    auto vertShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\skybox\\skyboxVert.spv");
    auto fragShaderCode = readFile(ExternalPath + "core\\graphics\\shaders\\skybox\\skyboxFrag.spv");
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


    std::array<VkPipelineColorBlendAttachmentState,4> colorBlendAttachment{};
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

    std::array<VkDescriptorSetLayout,1> SetLayouts = {DescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
    if (vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create Skybox pipeline layout!");

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
    if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create Skybox graphics pipeline!");

    vkDestroyShaderModule(*device, fragShaderModule, nullptr);
    vkDestroyShaderModule(*device, vertShaderModule, nullptr);
}

void deferredGraphics::createSkyboxDescriptorPool()
{
    size_t index = 0;

    std::array<VkDescriptorPoolSize,2> poolSizes;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(image.Count);
    index++;
        poolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.at(index).descriptorCount = static_cast<uint32_t>(image.Count);

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    if (vkCreateDescriptorPool(*device, &poolInfo, nullptr, &skybox.DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create Skybox descriptor pool!");
}

void deferredGraphics::createSkyboxDescriptorSets()
{
    skybox.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, skybox.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = skybox.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(*device, &allocInfo, skybox.DescriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate Skybox descriptor sets!");
}

void deferredGraphics::updateSkyboxDescriptorSets()
{
    for (size_t i = 0; i < image.Count; i++)
    {
        size_t index = 0;

        std::array<VkDescriptorBufferInfo,1> bufferInfo{};
            bufferInfo[0].buffer = skybox.uniformBuffers[i];
            bufferInfo[0].offset = 0;
            bufferInfo[0].range = sizeof(SkyboxUniformBufferObject);

        std::array<VkDescriptorImageInfo,1> skyboxImageInfo{};
            skyboxImageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            skyboxImageInfo[0].imageView = skybox.texture ? skybox.texture->getTextureImageView() : emptyTexture->getTextureImageView();
            skyboxImageInfo[0].sampler   = skybox.texture ? skybox.texture->getTextureSampler() : emptyTexture->getTextureSampler();

        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = skybox.DescriptorSets[i];
            descriptorWrites[index].dstBinding = 0;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[index].descriptorCount = static_cast<uint32_t>(bufferInfo.size());
            descriptorWrites[index].pBufferInfo = bufferInfo.data();
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = skybox.DescriptorSets[i];
            descriptorWrites[index].dstBinding = 1;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = static_cast<uint32_t>(skyboxImageInfo.size());
            descriptorWrites[index].pImageInfo = skyboxImageInfo.data();
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void deferredGraphics::Skybox::render(uint32_t frameNumber, VkCommandBuffer commandBuffers)
{
    if(objects.size()!=0)
    {
        vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
        vkCmdBindDescriptorSets(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &DescriptorSets[frameNumber], 0, NULL);
        vkCmdDraw(commandBuffers, 36, 1, 0, 0);
    }
}
