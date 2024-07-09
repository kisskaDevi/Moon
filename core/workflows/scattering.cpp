#include "scattering.h"
#include "vkdefault.h"
#include "light.h"
#include "operations.h"
#include "depthMap.h"

namespace moon::workflows {

struct ScatteringPushConst {
    alignas(4) uint32_t  width { 0 };
    alignas(4) uint32_t  height { 0 };
};

Scattering::Scattering(const moon::utils::ImageInfo& imageInfo, const std::filesystem::path& shadersPath, ScatteringParameters& parameters,
                       std::vector<moon::interfaces::Light*>* lightSources,
                       std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap>* depthMaps)
    : Workflow(imageInfo, shadersPath), parameters(parameters), lighting(this->imageInfo)
{
    lighting.lightSources = lightSources;
    lighting.depthMaps = depthMaps;
}

void Scattering::createAttachments(moon::utils::AttachmentsDatabase& aDatabase){
    moon::utils::createAttachments(physicalDevice, device, imageInfo, 1, &frame);
    aDatabase.addAttachmentData(parameters.out.scattering, parameters.enable, &frame);
}

void Scattering::createRenderPass()
{
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        moon::utils::Attachments::imageDescription(VK_FORMAT_R32G32B32A32_SFLOAT)
    };

    std::vector<std::vector<VkAttachmentReference>> attachmentRef;
    attachmentRef.push_back(std::vector<VkAttachmentReference>());
        attachmentRef.back().push_back(VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

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

void Scattering::createFramebuffers()
{
    framebuffers.resize(imageInfo.Count);
    for(size_t i = 0; i < imageInfo.Count; i++){
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &frame.imageView(i);
            framebufferInfo.width = imageInfo.Extent.width;
            framebufferInfo.height = imageInfo.Extent.height;
            framebufferInfo.layers = 1;
        framebuffers[i] = utils::vkDefault::Framebuffer(device, framebufferInfo);
    }
}

void Scattering::Lighting::create(const std::filesystem::path& vertShaderPath, const std::filesystem::path& fragShaderPath, VkDevice device, VkRenderPass pRenderPass)
{
    this->vertShaderPath = vertShaderPath;
    this->fragShaderPath = fragShaderPath;
    this->device = device;

    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(moon::utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);

    bufferDescriptorSetLayoutMap[moon::interfaces::LightType::spot] = moon::interfaces::Light::createBufferDescriptorSetLayout(device);
    descriptorSetLayoutMap[moon::interfaces::LightType::spot] = moon::interfaces::Light::createTextureDescriptorSetLayout(device);
    shadowDescriptorSetLayout = moon::utils::DepthMap::createDescriptorSetLayout(device);

    createPipeline(moon::interfaces::LightType::spot, pRenderPass);

    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageInfo.Count);
}

void Scattering::Lighting::createPipeline(uint8_t mask, VkRenderPass pRenderPass) {
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

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_MIN;
    VkPipelineColorBlendStateCreateInfo colorBlending = moon::utils::vkDefault::colorBlendState(1,&colorBlendAttachment);

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(ScatteringPushConst);
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
        descriptorSetLayout,
        shadowDescriptorSetLayout,
        bufferDescriptorSetLayoutMap[mask],
        descriptorSetLayoutMap[mask]
    };
    pipelineLayoutMap[mask] = utils::vkDefault::PipelineLayout(device, descriptorSetLayouts, pushConstantRange);

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
        pipelineInfo.back().layout = pipelineLayoutMap[mask];
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipelinesMap[mask] = utils::vkDefault::Pipeline(device, pipelineInfo);
}

void Scattering::create(moon::utils::AttachmentsDatabase& aDatabase)
{
    if(parameters.enable && !created){
        createAttachments(aDatabase);
        createRenderPass();
        createFramebuffers();
        lighting.create(shadersPath / "scattering/scatteringVert.spv", shadersPath / "scattering/scatteringFrag.spv", device, renderPass);
        created = true;
    }
}

void Scattering::updateDescriptorSets(
    const moon::utils::BuffersDatabase& bDatabase,
    const moon::utils::AttachmentsDatabase& aDatabase)
{
    if (!parameters.enable || !created) return;

    for (uint32_t i = 0; i < imageInfo.Count; i++)
    {
        VkDescriptorImageInfo depthInfos = aDatabase.descriptorImageInfo(parameters.in.depth, i);
        VkDescriptorBufferInfo bufferInfo = bDatabase.descriptorBufferInfo(parameters.in.camera, i);

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = lighting.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = lighting.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &depthInfos;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void Scattering::updateCommandBuffer(uint32_t frameNumber){
    if (!parameters.enable || !created) return;

    std::vector<VkClearValue> clearValues = {frame.clearValue()};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    for(auto& lightSource: *lighting.lightSources){
        if(lightSource->isScatteringEnable()){
            ScatteringPushConst pushConst{ imageInfo.Extent.width, imageInfo.Extent.height};
            uint8_t mask = lightSource->getPipelineBitMask();
            vkCmdPushConstants(commandBuffers[frameNumber], lighting.pipelineLayoutMap[lightSource->getPipelineBitMask()], VK_SHADER_STAGE_ALL, 0, sizeof(ScatteringPushConst), &pushConst);
            lightSource->render(frameNumber, commandBuffers[frameNumber], {lighting.descriptorSets[frameNumber], (*lighting.depthMaps)[lightSource].descriptorSets()[frameNumber]}, lighting.pipelineLayoutMap[mask], lighting.pipelinesMap[mask]);
        }
    }

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}

}
