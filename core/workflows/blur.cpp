#include "blur.h"
#include "operations.h"
#include "vkdefault.h"

namespace moon::workflows {

GaussianBlur::GaussianBlur(GaussianBlurParameters parameters, bool enable) :
    parameters(parameters), enable(enable)
{}

void GaussianBlur::createBufferAttachments(){
    bufferAttachment.create(physicalDevice,device,image.Format,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |VK_IMAGE_USAGE_SAMPLED_BIT,image.Extent,image.Count);
    VkSamplerCreateInfo samplerInfo = moon::utils::vkDefault::samler();
    CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &bufferAttachment.sampler));
}

void GaussianBlur::createAttachments(moon::utils::AttachmentsDatabase& aDatabase)
{
    createBufferAttachments();
    moon::utils::createAttachments(physicalDevice, device, image, 1, &frame);
    aDatabase.addAttachmentData(parameters.out.blur, enable, &frame);
}

void GaussianBlur::destroy(){
    xblur.destroy(device);
    yblur.destroy(device);

    Workflow::destroy();

    frame.deleteAttachment(device);
    frame.deleteSampler(device);

    bufferAttachment.deleteAttachment(device);
    bufferAttachment.deleteSampler(device);
}

void GaussianBlur::createRenderPass(){
    std::vector<VkAttachmentDescription> attachments = {
        moon::utils::Attachments::imageDescription(image.Format),
        moon::utils::Attachments::imageDescription(image.Format)
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

void GaussianBlur::createFramebuffers(){
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

void GaussianBlur::Blur::createDescriptorSetLayout(VkDevice device){
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    CHECK(descriptorSetLayout.create(device, bindings));
}

void GaussianBlur::createPipelines(){
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

void GaussianBlur::Blur::createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass){
    VkShaderModule vertShaderModule = moon::utils::shaderModule::create(&device, moon::utils::shaderModule::readFile(vertShaderPath));
    VkShaderModule fragShaderModule = moon::utils::shaderModule::create(&device, moon::utils::shaderModule::readFile(fragShaderPath));
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        moon::utils::vkDefault::vertrxShaderStage(vertShaderModule),
        moon::utils::vkDefault::fragmentShaderStage(fragShaderModule)
    };

    VkViewport viewport = moon::utils::vkDefault::viewport({0,0}, pInfo->Extent);
    VkRect2D scissor = moon::utils::vkDefault::scissor({0,0}, pInfo->Extent);
    VkPipelineViewportStateCreateInfo viewportState = moon::utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = moon::utils::vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = moon::utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = moon::utils::vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = moon::utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = moon::utils::vkDefault::depthStencilDisable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {moon::utils::vkDefault::colorBlendAttachmentState(VK_FALSE)};
    if(subpassNumber == 0){
        colorBlendAttachment.push_back(moon::utils::vkDefault::colorBlendAttachmentState(VK_FALSE));
    }
    VkPipelineColorBlendStateCreateInfo colorBlending = moon::utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
        pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(float);
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { descriptorSetLayout };
    CHECK(pipelineLayout.create(device, descriptorSetLayouts, pushConstantRange));

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
        pipelineInfo.back().layout = pipelineLayout;
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = subpassNumber;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    CHECK(pipeline.create(device, pipelineInfo));

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void GaussianBlur::createDescriptorPool(){
    Workflow::createDescriptorPool(device, &xblur, 0, image.Count, image.Count);
    Workflow::createDescriptorPool(device, &yblur, 0, image.Count, image.Count);
}

void GaussianBlur::createDescriptorSets(){
    Workflow::createDescriptorSets(device, &xblur, image.Count);
    Workflow::createDescriptorSets(device, &yblur, image.Count);
}

void GaussianBlur::create(moon::utils::AttachmentsDatabase& aDatabasep)
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

void GaussianBlur::updateDescriptorSets(
    const moon::utils::BuffersDatabase&,
    const moon::utils::AttachmentsDatabase& aDatabase)
{
    if(!enable) return;

    auto updateDescriptorSets = [](VkDevice device, const moon::utils::Attachments* attachment, VkSampler sampler, std::vector<VkDescriptorSet>& descriptorSets){
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

void GaussianBlur::updateCommandBuffer(uint32_t frameNumber){
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

        vkCmdPushConstants(commandBuffers[frameNumber], xblur.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(float), &blurDepth);

        vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, xblur.pipeline);
        vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, xblur.pipelineLayout, 0, 1, &xblur.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers[frameNumber], 6, 1, 0, 0);

    vkCmdNextSubpass(commandBuffers[frameNumber], VK_SUBPASS_CONTENTS_INLINE);
    vkCmdNextSubpass(commandBuffers[frameNumber], VK_SUBPASS_CONTENTS_INLINE);

        vkCmdPushConstants(commandBuffers[frameNumber], yblur.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(float), &blurDepth);

        vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, yblur.pipeline);
        vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, yblur.pipelineLayout, 0, 1, &yblur.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers[frameNumber], 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}

GaussianBlur& GaussianBlur::setBlurDepth(float blurDepth){
    this->blurDepth = blurDepth;
    return *this;
}

}
