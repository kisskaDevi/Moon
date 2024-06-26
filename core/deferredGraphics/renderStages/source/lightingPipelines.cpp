#include "graphics.h"
#include "operations.h"
#include "vkdefault.h"

#include <filesystem>

namespace moon::deferredGraphics {

void Graphics::Lighting::createPipeline(VkDevice device, uint8_t mask, VkRenderPass pRenderPass, std::filesystem::path vertShadersPath, std::filesystem::path fragShadersPath){
    uint8_t key = mask;

    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, vertShadersPath);
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, fragShadersPath);
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader, fragShader };

    VkViewport viewport = moon::utils::vkDefault::viewport({0,0}, imageInfo.Extent);
    VkRect2D scissor = moon::utils::vkDefault::scissor({0,0}, imageInfo.Extent);
    VkPipelineViewportStateCreateInfo viewportState = moon::utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = moon::utils::vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = moon::utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = moon::utils::vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = moon::utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = moon::utils::vkDefault::depthStencilDisable();

    VkPipelineColorBlendAttachmentState customBlendAttachment{};
        customBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        customBlendAttachment.blendEnable = VK_TRUE;
        customBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        customBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        customBlendAttachment.colorBlendOp = VK_BLEND_OP_MAX;
        customBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        customBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        customBlendAttachment.alphaBlendOp = VK_BLEND_OP_MIN;

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment ={
        moon::utils::vkDefault::colorBlendAttachmentState(VK_TRUE),
        customBlendAttachment,
        moon::utils::vkDefault::colorBlendAttachmentState(VK_TRUE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = moon::utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
        descriptorSetLayout,
        shadowDescriptorSetLayout,
        bufferDescriptorSetLayoutMap[key],
        textureDescriptorSetLayoutMap[key]
    };
    CHECK(pipelineLayoutMap[key].create(device, descriptorSetLayouts));

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
        pipelineInfo.back().layout = pipelineLayoutMap[key];
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = 1;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    CHECK(pipelineMap[key].create(device, pipelineInfo));
}

}
