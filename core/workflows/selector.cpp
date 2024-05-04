#include "selector.h"
#include "operations.h"
#include "vkdefault.h"

selectorGraphics::selectorGraphics(selectorParameters parameters, bool enable, uint32_t transparentLayersCount) :
    parameters(parameters), enable(enable)
{
    selector.transparentLayersCount = transparentLayersCount > 0 ? transparentLayersCount : 1;
}

void selectorGraphics::createAttachments(moon::utils::AttachmentsDatabase& aDatabase){
    moon::utils::createAttachments(physicalDevice, device, image, 1, &frame);
    aDatabase.addAttachmentData(parameters.out.selector, enable, &frame);
}

void selectorGraphics::destroy()
{
    selector.destroy(device);
    workflow::destroy();

    frame.deleteAttachment(device);
    frame.deleteSampler(device);
}

void selectorGraphics::createRenderPass()
{
    std::vector<VkAttachmentDescription> attachments = {
        moon::utils::Attachments::imageDescription(image.Format)
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

void selectorGraphics::createFramebuffers()
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

void selectorGraphics::createPipelines()
{
    selector.vertShaderPath = shadersPath / "selector/selectorVert.spv";
    selector.fragShaderPath = shadersPath / "selector/selectorFrag.spv";
    selector.createDescriptorSetLayout(device);
    selector.createPipeline(device,&image,renderPass);
}

void selectorGraphics::Selector::createDescriptorSetLayout(VkDevice device)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(VkDescriptorSetLayoutBinding{});
    bindings.back().binding = static_cast<uint32_t>(bindings.size() - 1);
    bindings.back().descriptorCount = 1;
    bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings.back().pImmutableSamplers = nullptr;
    bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), transparentLayersCount));
    bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), transparentLayersCount));

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &DescriptorSetLayout));
}

void selectorGraphics::Selector::createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass)
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

    auto vertShaderCode = moon::utils::shaderModule::readFile(vertShaderPath);
    auto fragShaderCode = moon::utils::shaderModule::readFile(fragShaderPath);
    VkShaderModule vertShaderModule = moon::utils::shaderModule::create(&device, vertShaderCode);
    VkShaderModule fragShaderModule = moon::utils::shaderModule::create(&device, fragShaderCode);
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        moon::utils::vkDefault::vertrxShaderStage(vertShaderModule),
        moon::utils::vkDefault::fragmentShaderStage(fragShaderModule)
    };
    shaderStages.back().pSpecializationInfo = &specializationInfo;

    VkViewport viewport = moon::utils::vkDefault::viewport({0,0}, pInfo->Extent);
    VkRect2D scissor = moon::utils::vkDefault::scissor({0,0}, pInfo->Extent);
    VkPipelineViewportStateCreateInfo viewportState = moon::utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = moon::utils::vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = moon::utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = moon::utils::vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = moon::utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = moon::utils::vkDefault::depthStencilDisable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {moon::utils::vkDefault::colorBlendAttachmentState(VK_FALSE)};
    VkPipelineColorBlendStateCreateInfo colorBlending = moon::utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &DescriptorSetLayout;
    CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &PipelineLayout));

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
    pipelineInfo.renderPass = pRenderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &depthStencil;
    CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Pipeline));

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void selectorGraphics::createDescriptorPool(){
    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, image.Count});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, image.Count});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, image.Count});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, image.Count * selector.transparentLayersCount});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, image.Count * selector.transparentLayersCount});

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = image.Count;
    CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &selector.DescriptorPool));
}

void selectorGraphics::createDescriptorSets(){
    workflow::createDescriptorSets(device, &selector, image.Count);
}

void selectorGraphics::create(moon::utils::AttachmentsDatabase& aDatabase)
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

void selectorGraphics::updateDescriptorSets(
    const moon::utils::BuffersDatabase& bDatabase,
    const moon::utils::AttachmentsDatabase& aDatabase)
{
    if(!enable) return;

    for (uint32_t i = 0; i < this->image.Count; i++)
    {
        VkDescriptorBufferInfo StorageBufferInfo = bDatabase.descriptorBufferInfo(parameters.in.storageBuffer, i);
        VkDescriptorImageInfo positionImageInfo = aDatabase.descriptorImageInfo(parameters.in.position, i);
        VkDescriptorImageInfo depthImageInfo = aDatabase.descriptorImageInfo(parameters.in.depth, i, parameters.in.defaultDepthTexture);

        std::vector<VkDescriptorImageInfo> positionLayersImageInfo(selector.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> depthLayersImageInfo(selector.transparentLayersCount);

        for(uint32_t index = 0; index < selector.transparentLayersCount; index++){
            std::string key = parameters.in.transparency + std::to_string(index) + ".";

            positionLayersImageInfo[index] = aDatabase.descriptorImageInfo(key + parameters.in.position, i);
            depthLayersImageInfo[index] = aDatabase.descriptorImageInfo(key + parameters.in.depth, i, parameters.in.defaultDepthTexture);
        }

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = selector.DescriptorSets.at(i);
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &StorageBufferInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = selector.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &positionImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = selector.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &depthImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = selector.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = selector.transparentLayersCount;
            descriptorWrites.back().pImageInfo = positionLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = selector.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = selector.transparentLayersCount;
            descriptorWrites.back().pImageInfo = depthLayersImageInfo.data();
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void selectorGraphics::updateCommandBuffer(uint32_t frameNumber){
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

    vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, selector.Pipeline);
    vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, selector.PipelineLayout, 0, 1, &selector.DescriptorSets[frameNumber], 0, nullptr);
    vkCmdDraw(commandBuffers[frameNumber], 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}
