#include "skybox.h"
#include "operations.h"
#include "vkdefault.h"

namespace moon::workflows {

SkyboxGraphics::SkyboxGraphics(SkyboxParameters& parameters, const interfaces::Objects* objects)
    : Workflow(parameters.imageInfo, parameters.shadersPath), parameters(parameters), skybox(parameters.imageInfo, objects)
{}

void SkyboxGraphics::createAttachments(moon::utils::AttachmentsDatabase& aDatabase){
    moon::utils::createAttachments(physicalDevice, device, parameters.imageInfo, 2, &frame, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, utils::vkDefault::sampler());
    aDatabase.addAttachmentData(parameters.out.baseColor, parameters.enable, &frame.color);
    aDatabase.addAttachmentData(parameters.out.bloom, parameters.enable, &frame.bloom);
}

void SkyboxGraphics::createRenderPass()
{
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        moon::utils::Attachments::imageDescription(parameters.imageInfo.Format),
        moon::utils::Attachments::imageDescription(parameters.imageInfo.Format)
    };

    std::vector<std::vector<VkAttachmentReference>> attachmentRef;
    attachmentRef.push_back(std::vector<VkAttachmentReference>());
        attachmentRef.back().push_back(VkAttachmentReference{static_cast<uint32_t>(attachmentRef.back().size()), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        attachmentRef.back().push_back(VkAttachmentReference{static_cast<uint32_t>(attachmentRef.back().size()), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    utils::vkDefault::RenderPass::SubpassDescriptions subpasses;
    for(auto refIt = attachmentRef.begin(); refIt != attachmentRef.end(); refIt++){
        subpasses.push_back(VkSubpassDescription{});
        subpasses.back().pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpasses.back().colorAttachmentCount = static_cast<uint32_t>(refIt->size());
        subpasses.back().pColorAttachments = refIt->data();
    }

    utils::vkDefault::RenderPass::SubpassDependencies dependencies;
    dependencies.push_back(VkSubpassDependency{});
        dependencies.back().srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies.back().dstSubpass = 0;
        dependencies.back().srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependencies.back().srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies.back().dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies.back().dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    renderPass = utils::vkDefault::RenderPass(device, attachments, subpasses, dependencies);
}

void SkyboxGraphics::createFramebuffers()
{
    framebuffers.resize(parameters.imageInfo.Count);
    for(size_t i = 0; i < parameters.imageInfo.Count; i++){
        std::vector<VkImageView> pAttachments = {
            frame.color.imageView(i),
            frame.bloom.imageView(i)
        };
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(pAttachments.size());
            framebufferInfo.pAttachments = pAttachments.data();
            framebufferInfo.width = parameters.imageInfo.Extent.width;
            framebufferInfo.height = parameters.imageInfo.Extent.height;
            framebufferInfo.layers = 1;
        framebuffers[i] = utils::vkDefault::Framebuffer(device, framebufferInfo);
    }
}

void SkyboxGraphics::Skybox::create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass) {
    this->vertShaderPath = vertShaderPath;
    this->fragShaderPath = fragShaderPath;
    this->device = device;

    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(moon::utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);

    objectDescriptorSetLayout = moon::interfaces::Object::createSkyboxDescriptorSetLayout(device);

    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, vertShaderPath);
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, fragShaderPath);
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader, fragShader };

    VkViewport viewport = moon::utils::vkDefault::viewport({0,0}, imageInfo.Extent);
    VkRect2D scissor = moon::utils::vkDefault::scissor({0,0}, imageInfo.Extent);
    VkPipelineViewportStateCreateInfo viewportState = moon::utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = moon::utils::vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = moon::utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = moon::utils::vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = moon::utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = moon::utils::vkDefault::depthStencilDisable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {
        moon::utils::vkDefault::colorBlendAttachmentState(VK_TRUE),
        moon::utils::vkDefault::colorBlendAttachmentState(VK_TRUE)};
    VkPipelineColorBlendStateCreateInfo colorBlending = moon::utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {descriptorSetLayout, objectDescriptorSetLayout};
    pipelineLayout = utils::vkDefault::PipelineLayout(device, descriptorSetLayouts);

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
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);

    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageInfo.Count);
}

void SkyboxGraphics::create(moon::utils::AttachmentsDatabase& aDatabase)
{
    if(parameters.enable && !created){
        createAttachments(aDatabase);
        createRenderPass();
        createFramebuffers();
        skybox.create(parameters.shadersPath / "skybox/skyboxVert.spv", parameters.shadersPath / "skybox/skyboxFrag.spv", device, renderPass);
        created = true;
    }
}

void SkyboxGraphics::updateDescriptorSets(const moon::utils::BuffersDatabase& bDatabase, const moon::utils::AttachmentsDatabase&) {
    if (!parameters.enable || !created) return;

    for (uint32_t i = 0; i < parameters.imageInfo.Count; i++){
        VkDescriptorBufferInfo bufferInfo = bDatabase.descriptorBufferInfo(parameters.in.camera, i);

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = skybox.descriptorSets[i];
            descriptorWrites.back().dstBinding = 0;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void SkyboxGraphics::updateCommandBuffer(uint32_t frameNumber){
    if (!parameters.enable || !created) return;

    std::vector<VkClearValue> clearValues = {frame.color.clearValue(), frame.bloom.clearValue()};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = parameters.imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipeline);
    for(const auto& object: *skybox.objects)
    {
        if((moon::interfaces::ObjectType::skybox & object->getPipelineBitMask()) && object->getEnable()){
            utils::vkDefault::DescriptorSets descriptorSets = {skybox.descriptorSets[frameNumber], object->getDescriptorSet(frameNumber)};
            vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipelineLayout, 0, static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, NULL);
            vkCmdDraw(commandBuffers[frameNumber], 36, 1, 0, 0);
        }
    }

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}

}
