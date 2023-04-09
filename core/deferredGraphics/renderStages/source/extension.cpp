#include "../graphics.h"
#include "../../../utils/operations.h"
#include "../../../utils/vkdefault.h"
#include "../../../transformational/object.h"
#include "../../../interfaces/model.h"

struct OutliningPushConst{
    alignas(16) glm::vec4           stencilColor;
    alignas(4)  float               width;
};

void graphics::OutliningExtension::DestroyPipeline(VkDevice device)
{
    if(Pipeline)         {vkDestroyPipeline(device, Pipeline, nullptr); Pipeline = VK_NULL_HANDLE;}
    if(PipelineLayout)   {vkDestroyPipelineLayout(device, PipelineLayout, nullptr); PipelineLayout = VK_NULL_HANDLE;}
}

void graphics::OutliningExtension::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass){
    auto vertShaderCode = ShaderModule::readFile(ExternalPath + "core\\deferredGraphics\\shaders\\outlining\\outliningVert.spv");
    auto fragShaderCode = ShaderModule::readFile(ExternalPath + "core\\deferredGraphics\\shaders\\outlining\\outliningFrag.spv");
    VkShaderModule vertShaderModule = ShaderModule::create(&device, vertShaderCode);
    VkShaderModule fragShaderModule = ShaderModule::create(&device, fragShaderCode);
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        vkDefault::vertrxShaderStage(vertShaderModule),
        vkDefault::fragmentShaderStage(fragShaderModule)
    };

    auto bindingDescription = model::Vertex::getBindingDescription();
    auto attributeDescriptions = model::Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkViewport viewport = vkDefault::viewport(pInfo->Extent);
    VkRect2D scissor = vkDefault::scissor(pInfo->Extent);
    VkPipelineViewportStateCreateInfo viewportState = vkDefault::viewportState(&viewport, &scissor);
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = vkDefault::rasterizationState(VK_FRONT_FACE_COUNTER_CLOCKWISE);
    VkPipelineMultisampleStateCreateInfo multisampling = vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = vkDefault::depthStencilDisable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {
        vkDefault::colorBlendAttachmentState(VK_FALSE),
        vkDefault::colorBlendAttachmentState(VK_FALSE),
        vkDefault::colorBlendAttachmentState(VK_FALSE),
        vkDefault::colorBlendAttachmentState(VK_FALSE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    struct PushConstBlock{ OutliningPushConst outlining; MaterialBlock material;};
    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(PushConstBlock);
    std::vector<VkDescriptorSetLayout> SetLayouts = {
        Parent->SceneDescriptorSetLayout,
        Parent->ObjectDescriptorSetLayout,
        Parent->PrimitiveDescriptorSetLayout,
        Parent->MaterialDescriptorSetLayout
    };
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = SetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &PipelineLayout);

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
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline);

    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
}

void graphics::OutliningExtension::render(uint32_t frameNumber, VkCommandBuffer commandBuffers)
{
    vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    for(auto object: Parent->objects){
        if(VkDeviceSize offsets = 0; object->getEnable()&&object->getOutliningEnable()){

            vkCmdBindVertexBuffers(commandBuffers, 0, 1, object->getModel()->getVertices(), &offsets);
            if (object->getModel()->getIndices() != VK_NULL_HANDLE){
                vkCmdBindIndexBuffer(commandBuffers, *object->getModel()->getIndices(), 0, VK_INDEX_TYPE_UINT32);
            }

            std::vector<VkDescriptorSet> descriptorSets = {Parent->DescriptorSets[frameNumber],object->getDescriptorSet()[frameNumber]};

            struct PushConstBlock{
                OutliningPushConst outlining;
                MaterialBlock material;
            } pushConstBlock;
                pushConstBlock.outlining.stencilColor = object->getOutliningColor();
                pushConstBlock.outlining.width = object->getOutliningWidth();

            uint32_t primirives = 0;
            object->getModel()->render(
                        object->getInstanceNumber(frameNumber),
                        commandBuffers,
                        PipelineLayout,
                        static_cast<uint32_t>(descriptorSets.size()),
                        descriptorSets.data(),
                        primirives,
                        sizeof(PushConstBlock),
                        offsetof(PushConstBlock, material),
                        &pushConstBlock);
        }
    }
}

