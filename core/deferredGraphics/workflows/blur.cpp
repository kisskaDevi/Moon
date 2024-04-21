#include "blur.h"
#include "operations.h"
#include "vkdefault.h"

gaussianBlur::gaussianBlur(gaussianBlurParameters parameters, bool enable) :
    parameters(parameters), enable(enable)
{}

void gaussianBlur::createBufferAttachments(){
    bufferAttachment.create(physicalDevice,device,image.Format,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |VK_IMAGE_USAGE_SAMPLED_BIT,image.Extent,image.Count);
    VkSamplerCreateInfo samplerInfo = vkDefault::samler();
    CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &bufferAttachment.sampler));
}

void gaussianBlur::createAttachments(attachmentsDatabase& aDatabase)
{
    createBufferAttachments();
    ::createAttachments(physicalDevice, device, image, 1, &frame);
    aDatabase.addAttachmentData(parameters.out.blur, enable, &frame);
}

void gaussianBlur::destroy(){
    xblur.destroy(device);
    yblur.destroy(device);

    workflow::destroy();

    frame.deleteAttachment(device);
    frame.deleteSampler(device);

    bufferAttachment.deleteAttachment(device);
    bufferAttachment.deleteSampler(device);
}

void gaussianBlur::createRenderPass(){
    std::vector<VkAttachmentDescription> attachments = {
        attachments::imageDescription(image.Format),
        attachments::imageDescription(image.Format)
    };

    std::vector<std::vector<VkAttachmentReference>> attachmentRef;
    attachmentRef.push_back({   VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
                                VkAttachmentReference{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}});
    attachmentRef.push_back({   VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}});
    attachmentRef.push_back({   VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}});

    std::vector<std::vector<VkAttachmentReference>> inAttachmentRef;
    inAttachmentRef.push_back(std::vector<VkAttachmentReference>());
    inAttachmentRef.push_back({ VkAttachmentReference{1,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}});
    inAttachmentRef.push_back(std::vector<VkAttachmentReference>());

    std::vector<VkSubpassDescription> subpass;
    for(auto refIt = attachmentRef.begin(), inRefIt = inAttachmentRef.begin();
        refIt != attachmentRef.end() && inRefIt != inAttachmentRef.end(); refIt++, inRefIt++){
        subpass.push_back(VkSubpassDescription{});
            subpass.back().pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.back().colorAttachmentCount = static_cast<uint32_t>(refIt->size());
            subpass.back().pColorAttachments = refIt->data();
            subpass.back().inputAttachmentCount = static_cast<uint32_t>(inRefIt->size());
            subpass.back().pInputAttachments = inRefIt->data();
    }

    std::vector<VkSubpassDependency> dependency;
    dependency.push_back(VkSubpassDependency{});
        dependency.back().srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.back().dstSubpass = 0;
        dependency.back().srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency.back().srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependency.back().dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.back().dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependency.push_back(VkSubpassDependency{});
        dependency.back().srcSubpass = 0;
        dependency.back().dstSubpass = 1;
        dependency.back().srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency.back().srcAccessMask = 0;
        dependency.back().dstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependency.back().dstAccessMask = 0;
    dependency.push_back(VkSubpassDependency{});
        dependency.back().srcSubpass = 1;
        dependency.back().dstSubpass = 2;
        dependency.back().srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.back().srcAccessMask = 0;
        dependency.back().dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.back().dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

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

void gaussianBlur::createFramebuffers(){
    for (auto attIt = frame.instances.begin(), buffIt = bufferAttachment.instances.begin();
         attIt != frame.instances.end() && buffIt != bufferAttachment.instances.end();
         attIt++, buffIt++)
    {
        std::vector<VkImageView> attachments = {attIt->imageView, buffIt->imageView};
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        framebuffers.push_back(VkFramebuffer{});
        CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers.back()));
    }
}

void gaussianBlur::blur::createDescriptorSetLayout(VkDevice device){
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &DescriptorSetLayout));
}

void gaussianBlur::createPipelines(){
    xblur.subpassNumber = 0;
    xblur.vertShaderPath = shadersPath / "gaussianBlur/xBlurVert.spv";
    xblur.fragShaderPath = shadersPath / "gaussianBlur/xBlurFrag.spv";
    xblur.createDescriptorSetLayout(device);
    xblur.createPipeline(device,&image,renderPass);

    yblur.subpassNumber = 2;
    yblur.vertShaderPath = shadersPath / "gaussianBlur/yBlurVert.spv";
    yblur.fragShaderPath = shadersPath / "gaussianBlur/yBlurFrag.spv";
    yblur.createDescriptorSetLayout(device);
    yblur.createPipeline(device,&image,renderPass);
}

void gaussianBlur::blur::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass){
    VkShaderModule vertShaderModule = ShaderModule::create(&device, ShaderModule::readFile(vertShaderPath));
    VkShaderModule fragShaderModule = ShaderModule::create(&device, ShaderModule::readFile(fragShaderPath));
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

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {vkDefault::colorBlendAttachmentState(VK_FALSE)};
    if(subpassNumber == 0){
        colorBlendAttachment.push_back(vkDefault::colorBlendAttachmentState(VK_FALSE));
    }
    VkPipelineColorBlendStateCreateInfo colorBlending = vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
        pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(float);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &DescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &PipelineLayout));

    std::vector<VkGraphicsPipelineCreateInfo> pipelineInfo;
    pipelineInfo.push_back(VkGraphicsPipelineCreateInfo{});
        pipelineInfo.back().sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.back().pNext = nullptr;
        pipelineInfo.back().stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.back().pStages = shaderStages.data();
        pipelineInfo.back().pVertexInputState = &vertexInputInfo;
        pipelineInfo.back().pInputAssemblyState = &inputAssembly;
        pipelineInfo.back().pViewportState = &viewportState;
        pipelineInfo.back().pRasterizationState = &rasterizer;
        pipelineInfo.back().pMultisampleState = &multisampling;
        pipelineInfo.back().pColorBlendState = &colorBlending;
        pipelineInfo.back().layout = PipelineLayout;
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = subpassNumber;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline));

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void gaussianBlur::createDescriptorPool(){
    workflow::createDescriptorPool(device, &xblur, 0, image.Count, image.Count);
    workflow::createDescriptorPool(device, &yblur, 0, image.Count, image.Count);
}

void gaussianBlur::createDescriptorSets(){
    workflow::createDescriptorSets(device, &xblur, image.Count);
    workflow::createDescriptorSets(device, &yblur, image.Count);
}

void gaussianBlur::create(attachmentsDatabase& aDatabasep)
{
    if(enable){
        createAttachments(aDatabasep);
        createRenderPass();
        createFramebuffers();
        createPipelines();
        createDescriptorPool();
        createDescriptorSets();
    }
}

void gaussianBlur::updateDescriptorSets(
    const buffersDatabase&,
    const attachmentsDatabase& aDatabase)
{
    if(!enable) return;

    auto updateDescriptorSets = [](VkDevice device, const attachments* attachment, VkSampler sampler, std::vector<VkDescriptorSet>& descriptorSets){
        auto imageIt = attachment->instances.begin();
        auto setIt = descriptorSets.begin();
        for (;imageIt != attachment->instances.end() && setIt != descriptorSets.end(); imageIt++, setIt++){
            VkDescriptorImageInfo imageInfo{};
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo.imageView = imageIt->imageView;
                imageInfo.sampler = sampler;

            std::vector<VkWriteDescriptorSet> descriptorWrites;
            descriptorWrites.push_back(VkWriteDescriptorSet{});
                descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites.back().dstSet = *setIt;
                descriptorWrites.back().dstBinding = static_cast<uint32_t>(static_cast<uint32_t>(descriptorWrites.size() - 1));
                descriptorWrites.back().dstArrayElement = 0;
                descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                descriptorWrites.back().descriptorCount = 1;
                descriptorWrites.back().pImageInfo = &imageInfo;
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    };

    const auto blurAttachment = aDatabase.get(parameters.in.blur);
    updateDescriptorSets(device, blurAttachment, blurAttachment->sampler, xblur.DescriptorSets);
    updateDescriptorSets(device, &bufferAttachment, bufferAttachment.sampler, yblur.DescriptorSets);
}

void gaussianBlur::updateCommandBuffer(uint32_t frameNumber){
    if(!enable) return;

    std::vector<VkClearValue> clearValues(2, VkClearValue{frame.clearValue.color});

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdPushConstants(commandBuffers[frameNumber], xblur.PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(float), &blurDepth);

        vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, xblur.Pipeline);
        vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, xblur.PipelineLayout, 0, 1, &xblur.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers[frameNumber], 6, 1, 0, 0);

    vkCmdNextSubpass(commandBuffers[frameNumber], VK_SUBPASS_CONTENTS_INLINE);
    vkCmdNextSubpass(commandBuffers[frameNumber], VK_SUBPASS_CONTENTS_INLINE);

        vkCmdPushConstants(commandBuffers[frameNumber], yblur.PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(float), &blurDepth);

        vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, yblur.Pipeline);
        vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, yblur.PipelineLayout, 0, 1, &yblur.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers[frameNumber], 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}

gaussianBlur& gaussianBlur::setBlurDepth(float blurDepth){
    this->blurDepth = blurDepth;
    return *this;
}
