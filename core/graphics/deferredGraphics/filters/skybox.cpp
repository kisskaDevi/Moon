#include "skybox.h"
#include "core/operations.h"
#include "core/transformational/object.h"
#include "core/transformational/camera.h"

#include <iostream>

skyboxGraphics::skyboxGraphics()
{

}

void skyboxGraphics::setEmptyTexture(texture* emptyTexture)
{
    this->emptyTexture = emptyTexture;
}

void skyboxGraphics::setExternalPath(const std::string &path)
{
    skybox.ExternalPath = path;
}

void skyboxGraphics::setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device)
{
    this->physicalDevice = physicalDevice;
    this->device = device;
}
void skyboxGraphics::setImageProp(imageInfo* pInfo) {this->image = *pInfo;}
void skyboxGraphics::setAttachments(uint32_t attachmentsCount, attachments* pAttachments)
{
    this->attachmentsCount = attachmentsCount;
    this->pAttachments = pAttachments;
}

void skyboxGraphics::createAttachments(uint32_t attachmentsCount, attachments* pAttachments)
{
    for(size_t attachmentNumber=0; attachmentNumber<attachmentsCount; attachmentNumber++)
    {
        pAttachments[attachmentNumber].create(physicalDevice,device,image.Format,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,image.Extent,image.Count);
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
        vkCreateSampler(*device, &samplerInfo, nullptr, &pAttachments[attachmentNumber].sampler);
    }
}

void skyboxGraphics::Skybox::Destroy(VkDevice* device)
{
    if(Pipeline)                    {vkDestroyPipeline(*device, Pipeline, nullptr); Pipeline = VK_NULL_HANDLE;}
    if(PipelineLayout)              {vkDestroyPipelineLayout(*device, PipelineLayout,nullptr); PipelineLayout = VK_NULL_HANDLE;}
    if(DescriptorSetLayout)         {vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr); DescriptorSetLayout = VK_NULL_HANDLE;}
    if(ObjectDescriptorSetLayout)   {vkDestroyDescriptorSetLayout(*device, ObjectDescriptorSetLayout, nullptr); ObjectDescriptorSetLayout = VK_NULL_HANDLE;}
    if(DescriptorPool)              {vkDestroyDescriptorPool(*device, DescriptorPool, nullptr); DescriptorPool = VK_NULL_HANDLE;}
}

void skyboxGraphics::destroy()
{
    skybox.Destroy(device);

    if(renderPass) {vkDestroyRenderPass(*device, renderPass, nullptr); renderPass = VK_NULL_HANDLE;}
    for(size_t i = 0; i< framebuffers.size();i++)
        if(framebuffers[i]) vkDestroyFramebuffer(*device, framebuffers[i],nullptr);
    framebuffers.resize(0);
}

void skyboxGraphics::createRenderPass()
{
    uint32_t index = 0;
    std::array<VkAttachmentDescription,1> attachments{};
        attachments[index] = attachments::imageDescription(image.Format);

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

void skyboxGraphics::createFramebuffers()
{
    framebuffers.resize(image.Count);
    for(size_t i = 0; i < image.Count; i++){
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &pAttachments->imageView[i];
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &framebuffers[i]);
    }
}

void skyboxGraphics::createPipelines()
{
    skybox.createDescriptorSetLayout(device);
    skybox.createPipeline(device,&image,&renderPass);
}

void skyboxGraphics::Skybox::createDescriptorSetLayout(VkDevice* device)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(VkDescriptorSetLayoutBinding{});
        bindings.back().binding = 0;
        bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings.back().descriptorCount = 1;
        bindings.back().stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        bindings.back().pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(*device, &layoutInfo, nullptr, &DescriptorSetLayout);

    skyboxObject::createDescriptorSetLayout(*device,&ObjectDescriptorSetLayout);
}

void skyboxGraphics::Skybox::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    uint32_t index = 0;

    auto vertShaderCode =ShaderModule::readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\skybox\\skyboxVert.spv");
    auto fragShaderCode =ShaderModule::readFile(ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\skybox\\skyboxFrag.spv");
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


    std::array<VkPipelineColorBlendAttachmentState,1> colorBlendAttachment{};
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

    std::array<VkDescriptorSetLayout,2> SetLayouts = {DescriptorSetLayout, ObjectDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
    vkCreatePipelineLayout(*device, &pipelineLayoutInfo, nullptr, &PipelineLayout);

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
    std::array<VkGraphicsPipelineCreateInfo,1> pipelineInfo{};
        pipelineInfo[index].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo[index].pNext = nullptr;
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

void skyboxGraphics::createDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    vkCreateDescriptorPool(*device, &poolInfo, nullptr, &skybox.DescriptorPool);
}

void skyboxGraphics::createDescriptorSets()
{
    skybox.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, skybox.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = skybox.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(*device, &allocInfo, skybox.DescriptorSets.data());
}

void skyboxGraphics::updateDescriptorSets(camera* cameraObject)
{
    for (size_t i = 0; i < image.Count; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = cameraObject->getBuffer(i);
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = skybox.DescriptorSets[i];
            descriptorWrites.back().dstBinding = 0;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void skyboxGraphics::createCommandBuffers(VkCommandPool commandPool)
{
    commandBuffers.resize(image.Count);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(image.Count);
    vkAllocateCommandBuffers(*device, &allocInfo, commandBuffers.data());
}

void skyboxGraphics::beginCommandBuffer(uint32_t frameNumber){
    vkResetCommandBuffer(commandBuffers[frameNumber],0);

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

    vkBeginCommandBuffer(commandBuffers[frameNumber], &beginInfo);
}

void skyboxGraphics::endCommandBuffer(uint32_t frameNumber){
    vkEndCommandBuffer(commandBuffers[frameNumber]);
}

void skyboxGraphics::updateCommandBuffer(uint32_t frameNumber)
{
        std::array<VkClearValue, 1> ClearValues{};
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

        vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.Pipeline);
        for(auto object: skybox.objects)
        {
            if(object->getEnable()){
                std::vector<VkDescriptorSet> descriptorSets = {skybox.DescriptorSets[frameNumber],object->getDescriptorSet()[frameNumber]};
                vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.PipelineLayout, 0, static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, NULL);
                vkCmdDraw(commandBuffers[frameNumber], 36, 1, 0, 0);
            }
        }

        vkCmdEndRenderPass(commandBuffers[frameNumber]);
}

VkCommandBuffer& skyboxGraphics::getCommandBuffer(uint32_t frameNumber)
{
    return commandBuffers[frameNumber];
}

void skyboxGraphics::bindObject(skyboxObject* newObject)
{
    skybox.objects.push_back(newObject);
}

bool skyboxGraphics::removeObject(skyboxObject* object)
{
    bool result = false;
    for(uint32_t index = 0; index<skybox.objects.size(); index++){
        if(object==skybox.objects[index]){
            skybox.objects.erase(skybox.objects.begin()+index);
            result = true;
        }
    }
    return result;
}

void skyboxGraphics::updateObjectUniformBuffer(VkCommandBuffer commandBuffer, uint32_t currentImage)
{
    for(size_t i=0;i<skybox.objects.size();i++)
        skybox.objects[i]->updateUniformBuffer(commandBuffer, currentImage);
}
