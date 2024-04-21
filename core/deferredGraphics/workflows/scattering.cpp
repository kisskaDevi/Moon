#include "scattering.h"
#include "vkdefault.h"
#include "light.h"
#include "operations.h"
#include "depthMap.h"

scattering::scattering(scatteringParameters parameters,
                       bool enable, std::vector<light*>* lightSources,
                       std::unordered_map<light*, depthMap*>* depthMaps) :
    parameters(parameters), enable(enable)
{
    lighting.lightSources = lightSources;
    lighting.depthMaps = depthMaps;
}

void scattering::Lighting::destroy(VkDevice device){
    workbody::destroy(device);
    if(ShadowDescriptorSetLayout) {vkDestroyDescriptorSetLayout(device, ShadowDescriptorSetLayout, nullptr); ShadowDescriptorSetLayout = VK_NULL_HANDLE;}
    for(auto& descriptorSetLayout: BufferDescriptorSetLayoutDictionary){
        if(descriptorSetLayout.second){ vkDestroyDescriptorSetLayout(device, descriptorSetLayout.second, nullptr); descriptorSetLayout.second = VK_NULL_HANDLE;}
    }
    for(auto& descriptorSetLayout: DescriptorSetLayoutDictionary){
        if(descriptorSetLayout.second){ vkDestroyDescriptorSetLayout(device, descriptorSetLayout.second, nullptr); descriptorSetLayout.second = VK_NULL_HANDLE;}
    }
    for(auto& PipelineLayout: PipelineLayoutDictionary){
        if(PipelineLayout.second) {
            vkDestroyPipelineLayout(device, PipelineLayout.second, nullptr);
            PipelineLayout.second = VK_NULL_HANDLE;}
    }
    for(auto& Pipeline: PipelinesDictionary){
        if(Pipeline.second) {
            vkDestroyPipeline(device, Pipeline.second, nullptr);
            Pipeline.second = VK_NULL_HANDLE;}
    }
}

void scattering::createAttachments(attachmentsDatabase& aDatabase){
    ::createAttachments(physicalDevice, device, image, 1, &frame);
    aDatabase.addAttachmentData(parameters.out.scattering, enable, &frame);
}

void scattering::destroy()
{
    lighting.destroy(device);
    workflow::destroy();

    frame.deleteAttachment(device);
    frame.deleteSampler(device);
}

void scattering::createRenderPass()
{
    std::vector<VkAttachmentDescription> attachments = {
        attachments::imageDescription(VK_FORMAT_R32G32B32A32_SFLOAT)
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
    CHECK(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
}

void scattering::createFramebuffers()
{
    framebuffers.resize(image.Count);
    for(size_t i = 0; i < image.Count; i++){
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &frame.instances[i].imageView;
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[i]));
    }
}

void scattering::createPipelines()
{
    lighting.vertShaderPath = shadersPath / "scattering/scatteringVert.spv";
    lighting.fragShaderPath = shadersPath / "scattering/scatteringFrag.spv";
    lighting.createDescriptorSetLayout(device);
    lighting.createPipeline(device,&image,renderPass);
}

void scattering::Lighting::createDescriptorSetLayout(VkDevice device)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &DescriptorSetLayout));

    light::createBufferDescriptorSetLayout(device,&BufferDescriptorSetLayoutDictionary[lightType::spot]);
    light::createTextureDescriptorSetLayout(device,&DescriptorSetLayoutDictionary[lightType::spot]);
    depthMap::createDescriptorSetLayout(device, &ShadowDescriptorSetLayout);
}

void scattering::Lighting::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass)
{
    createPipeline(lightType::spot, device, pInfo, pRenderPass);
}

void scattering::Lighting::createPipeline(uint8_t mask, VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass)
{
    auto vertShaderCode = ShaderModule::readFile(vertShaderPath);
    auto fragShaderCode = ShaderModule::readFile(fragShaderPath);
    VkShaderModule vertShaderModule = ShaderModule::create(&device, vertShaderCode);
    VkShaderModule fragShaderModule = ShaderModule::create(&device, fragShaderCode);
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        vkDefault::vertrxShaderStage(vertShaderModule),
        vkDefault::fragmentShaderStage(fragShaderModule)
    };

    VkViewport viewport = vkDefault::viewport({0,0}, pInfo->Extent);
    VkRect2D scissor = vkDefault::scissor({0,0}, pInfo->Extent);
    VkPipelineViewportStateCreateInfo viewportState = vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = vkDefault::depthStencilDisable();

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_MIN;
    VkPipelineColorBlendStateCreateInfo colorBlending = vkDefault::colorBlendState(1,&colorBlendAttachment);

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(scatteringPushConst);
    std::vector<VkDescriptorSetLayout> SetLayouts = {
        DescriptorSetLayout,
        ShadowDescriptorSetLayout,
        BufferDescriptorSetLayoutDictionary[mask],
        DescriptorSetLayoutDictionary[mask]
    };
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &PipelineLayoutDictionary[mask]));

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
        pipelineInfo.layout = PipelineLayoutDictionary[mask];
        pipelineInfo.renderPass = pRenderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.pDepthStencilState = &depthStencil;
    CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &PipelinesDictionary[mask]));

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void scattering::createDescriptorPool(){
    workflow::createDescriptorPool(device, &lighting, image.Count, image.Count, image.Count);
}

void scattering::createDescriptorSets(){
    workflow::createDescriptorSets(device, &lighting, image.Count);
}

void scattering::create(attachmentsDatabase& aDatabase)
{
    if(enable){
        createAttachments(aDatabase);
        createRenderPass();
        createFramebuffers();
        createPipelines();
        createDescriptorPool();
        createDescriptorSets();
    }
}

void scattering::updateDescriptorSets(
    const buffersDatabase& bDatabase,
    const attachmentsDatabase& aDatabase)
{
    if(!enable) return;

    for (uint32_t i = 0; i < image.Count; i++)
    {
        VkDescriptorImageInfo depthInfos = aDatabase.descriptorImageInfo(parameters.in.depth, i);
        VkDescriptorBufferInfo bufferInfo = bDatabase.descriptorBufferInfo(parameters.in.camera, i);

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = lighting.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = lighting.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &depthInfos;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void scattering::updateCommandBuffer(uint32_t frameNumber){
    if(!enable) return;

    std::vector<VkClearValue> clearValues = {frame.clearValue};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    for(auto& lightSource: *lighting.lightSources){
        if(lightSource->isScatteringEnable()){
            scatteringPushConst pushConst{image.Extent.width, image.Extent.height};
            uint8_t mask = lightSource->getPipelineBitMask();
            vkCmdPushConstants(commandBuffers[frameNumber], lighting.PipelineLayoutDictionary[lightSource->getPipelineBitMask()], VK_SHADER_STAGE_ALL, 0, sizeof(scatteringPushConst), &pushConst);
            lightSource->render(frameNumber, commandBuffers[frameNumber], {lighting.DescriptorSets[frameNumber], (*lighting.depthMaps)[lightSource]->getDescriptorSets()[frameNumber]}, lighting.PipelineLayoutDictionary[mask], lighting.PipelinesDictionary[mask]);
        }
    }

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}
