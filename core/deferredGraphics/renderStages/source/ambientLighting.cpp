#include "../graphics.h"
#include "operations.h"

namespace moon::deferredGraphics {

struct LightPassPushConst{
    alignas(4) float minAmbientFactor;
};

void Graphics::AmbientLighting::createPipeline(VkDevice device, moon::utils::ImageInfo* pInfo, VkRenderPass pRenderPass){
    auto vertShaderCode = moon::utils::shaderModule::readFile(shadersPath / "ambientLightingPass/ambientLightingVert.spv");
    auto fragShaderCode = moon::utils::shaderModule::readFile(shadersPath / "ambientLightingPass/ambientLightingFrag.spv");
    VkShaderModule vertShaderModule = moon::utils::shaderModule::create(&device, vertShaderCode);
    VkShaderModule fragShaderModule = moon::utils::shaderModule::create(&device, fragShaderCode);
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

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment ={
        moon::utils::vkDefault::colorBlendAttachmentState(VK_TRUE),
        moon::utils::vkDefault::colorBlendAttachmentState(VK_TRUE),
        moon::utils::vkDefault::colorBlendAttachmentState(VK_TRUE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = moon::utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(LightPassPushConst);
    std::vector<VkDescriptorSetLayout> descriptorSetLayout = {
        parent->lightingDescriptorSetLayout
    };
    CHECK(pipelineLayout.create(device, descriptorSetLayout, pushConstantRange));

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
        pipelineInfo.back().subpass = 1;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    CHECK(pipeline.create(device, pipelineInfo));

    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
}

void Graphics::AmbientLighting::render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const {
    LightPassPushConst pushConst{ minAmbientFactor };
    vkCmdPushConstants(commandBuffers, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(LightPassPushConst), &pushConst);

    vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &parent->DescriptorSets[frameNumber], 0, nullptr);
    vkCmdDraw(commandBuffers, 6, 1, 0, 0);
}

}
