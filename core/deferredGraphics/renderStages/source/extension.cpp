#include "../graphics.h"
#include "operations.h"
#include "vkdefault.h"
#include "object.h"
#include "model.h"
#include "vector.h"

namespace moon::deferredGraphics {

namespace{
struct OutliningPushConstBlock {
    struct {
        alignas(16) moon::math::Vector<float, 4> stencilColor;
        alignas(4)  float width;
    } outlining;
    moon::interfaces::MaterialBlock material;
};
}

Graphics::OutliningExtension::OutliningExtension(const Graphics::Base& parent)
    : parent(parent)
{}

void Graphics::OutliningExtension::create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass){
    this->shadersPath = shadersPath;

    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, shadersPath / "outlining/outliningVert.spv");
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, shadersPath / "outlining/outliningFrag.spv");
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader, fragShader };

    auto bindingDescription = moon::interfaces::Model::Vertex::getBindingDescription();
    auto attributeDescriptions = moon::interfaces::Model::Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkViewport viewport = moon::utils::vkDefault::viewport({0,0}, parent.imageInfo.Extent);
    VkRect2D scissor = moon::utils::vkDefault::scissor({0,0}, parent.imageInfo.Extent);
    VkPipelineViewportStateCreateInfo viewportState = moon::utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = moon::utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = moon::utils::vkDefault::rasterizationState(VK_FRONT_FACE_COUNTER_CLOCKWISE);
    VkPipelineMultisampleStateCreateInfo multisampling = moon::utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = moon::utils::vkDefault::depthStencilDisable();

    depthStencil.stencilTestEnable = VK_TRUE;
    depthStencil.back.compareOp = VK_COMPARE_OP_NOT_EQUAL;
    depthStencil.back.failOp = VK_STENCIL_OP_KEEP;
    depthStencil.back.depthFailOp = VK_STENCIL_OP_KEEP;
    depthStencil.back.passOp = VK_STENCIL_OP_REPLACE;
    depthStencil.back.compareMask = 0xff;
    depthStencil.back.writeMask = 0xff;
    depthStencil.back.reference = 1;
    depthStencil.front = depthStencil.back;

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {
        moon::utils::vkDefault::colorBlendAttachmentState(VK_FALSE),
        moon::utils::vkDefault::colorBlendAttachmentState(VK_FALSE),
        moon::utils::vkDefault::colorBlendAttachmentState(VK_FALSE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = moon::utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(OutliningPushConstBlock);
    std::vector<VkDescriptorSetLayout> descriptorSetLayout = {
        parent.descriptorSetLayout,
        parent.objectDescriptorSetLayout,
        parent.primitiveDescriptorSetLayout,
        parent.materialDescriptorSetLayout
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
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    CHECK(pipeline.create(device, pipelineInfo));
}

void Graphics::OutliningExtension::render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const
{
    vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    for(const auto& object: *parent.objects){
        if(VkDeviceSize offsets = 0; (moon::interfaces::ObjectType::base & object->getPipelineBitMask()) && object->getEnable() && object->getOutliningEnable()){
            vkCmdBindVertexBuffers(commandBuffers, 0, 1, object->getModel()->getVertices(), &offsets);
            if (object->getModel()->getIndices() != VK_NULL_HANDLE){
                vkCmdBindIndexBuffer(commandBuffers, *object->getModel()->getIndices(), 0, VK_INDEX_TYPE_UINT32);
            }

            utils::vkDefault::DescriptorSets descriptorSets = {parent.descriptorSets[frameNumber], object->getDescriptorSet(frameNumber)};

            OutliningPushConstBlock pushConstBlock;
            pushConstBlock.outlining.stencilColor = object->getOutliningColor();
            pushConstBlock.outlining.width = object->getOutliningWidth();

            uint32_t primirives = 0;
            object->getModel()->render(
                        object->getInstanceNumber(frameNumber),
                        commandBuffers,
                        pipelineLayout,
                        static_cast<uint32_t>(descriptorSets.size()),
                        descriptorSets.data(),
                        primirives,
                        sizeof(OutliningPushConstBlock),
                        offsetof(OutliningPushConstBlock, material),
                        &pushConstBlock);
        }
    }
}

}
