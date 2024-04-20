#include "boundingBox.h"
#include "vkdefault.h"
#include "operations.h"

#include "object.h"
#include "model.h"

namespace {
    struct PushConstBlock {
        alignas(16) vector<float,3> min;
        alignas(16) vector<float,3> max;
    };
}

boundingBoxGraphics::boundingBoxGraphics(bool enable, std::vector<object*>* objects) :
    enable(enable)
{
    box.objects = objects;
}

void boundingBoxGraphics::boundingBox::destroy(VkDevice device){
    workbody::destroy(device);
    if(ObjectDescriptorSetLayout)       {vkDestroyDescriptorSetLayout(device, ObjectDescriptorSetLayout, nullptr); ObjectDescriptorSetLayout = VK_NULL_HANDLE;}
    if(PrimitiveDescriptorSetLayout)    {vkDestroyDescriptorSetLayout(device, PrimitiveDescriptorSetLayout, nullptr); PrimitiveDescriptorSetLayout = VK_NULL_HANDLE;}
}

void boundingBoxGraphics::destroy(){
    box.destroy(device);
    workflow::destroy();

    frame.deleteAttachment(device);
    frame.deleteSampler(device);
}

void boundingBoxGraphics::createAttachments(attachmentsDatabase& aDatabase){
    ::createAttachments(physicalDevice, device, image, 1, &frame);
    aDatabase.addAttachmentData("boundingBox", enable, &frame);
}

void boundingBoxGraphics::createRenderPass(){
    std::vector<VkAttachmentDescription> attachments = {
        attachments::imageDescription(image.Format)
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

void boundingBoxGraphics::createFramebuffers(){
    framebuffers.resize(image.Count);
    for(size_t i = 0; i < image.Count; i++){
        std::vector<VkImageView> pAttachments = {frame.instances[i].imageView};
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(pAttachments.size());
        framebufferInfo.pAttachments = pAttachments.data();
        framebufferInfo.width = image.Extent.width;
        framebufferInfo.height = image.Extent.height;
        framebufferInfo.layers = 1;
        CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[i]));
    }
}

void boundingBoxGraphics::createPipelines(){
    box.vertShaderPath = shadersPath / "boundingBox/boundingBoxVert.spv";
    box.fragShaderPath = shadersPath / "boundingBox/boundingBoxFrag.spv";
    box.createDescriptorSetLayout(device);
    box.createPipeline(device,&image,renderPass);
}

void boundingBoxGraphics::boundingBox::createDescriptorSetLayout(VkDevice device){
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &DescriptorSetLayout));

    object::createDescriptorSetLayout(device,&ObjectDescriptorSetLayout);
    model::createNodeDescriptorSetLayout(device,&PrimitiveDescriptorSetLayout);
}

void boundingBoxGraphics::boundingBox::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass){
    auto vertShaderCode = ShaderModule::readFile(vertShaderPath);
    auto fragShaderCode = ShaderModule::readFile(fragShaderPath);
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

    VkViewport viewport = vkDefault::viewport({0,0}, pInfo->Extent);
    VkRect2D scissor = vkDefault::scissor({0,0}, pInfo->Extent);
    VkPipelineViewportStateCreateInfo viewportState = vkDefault::viewportState(&viewport, &scissor);
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = vkDefault::rasterizationState(VK_FRONT_FACE_COUNTER_CLOCKWISE);
    VkPipelineMultisampleStateCreateInfo multisampling = vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = vkDefault::depthStencilDisable();

    rasterizer.polygonMode = VK_POLYGON_MODE_LINE;

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {
        vkDefault::colorBlendAttachmentState(VK_FALSE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
    pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
    pushConstantRange.back().offset = 0;
    pushConstantRange.back().size = sizeof(PushConstBlock);
    std::vector<VkDescriptorSetLayout> setLayouts = {
        DescriptorSetLayout,
        ObjectDescriptorSetLayout,
        PrimitiveDescriptorSetLayout
    };
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pipelineLayoutInfo.pSetLayouts = setLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
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
    pipelineInfo.back().subpass = 0;
    pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &Pipeline));

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void boundingBoxGraphics::createDescriptorPool(){
    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &box.DescriptorPool);
}

void boundingBoxGraphics::createDescriptorSets(){
    box.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, box.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = box.DescriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
    allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(device, &allocInfo, box.DescriptorSets.data());
}

void boundingBoxGraphics::create(attachmentsDatabase& aDatabase)
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

void boundingBoxGraphics::updateDescriptorSets(
    const buffersDatabase& bDatabase,
    const attachmentsDatabase&)
{
    if(!enable) return;

    for (uint32_t i = 0; i < image.Count; i++)
    {
        VkDescriptorBufferInfo bufferInfo = bDatabase.descriptorBufferInfo("camera", i);

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = box.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void boundingBoxGraphics::updateCommandBuffer(uint32_t frameNumber){
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

        box.render(frameNumber, commandBuffers[frameNumber]);

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}

void boundingBoxGraphics::boundingBox::render(uint32_t frameNumber, VkCommandBuffer commandBuffers){
    for(auto object: *objects){
        if(VkDeviceSize offsets = 0; (objectType::base & object->getPipelineBitMask()) && object->getEnable()){
            vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);

            vkCmdBindVertexBuffers(commandBuffers, 0, 1, object->getModel()->getVertices(), &offsets);
            if (object->getModel()->getIndices() != VK_NULL_HANDLE){
                vkCmdBindIndexBuffer(commandBuffers, *object->getModel()->getIndices(), 0, VK_INDEX_TYPE_UINT32);
            }

            std::vector<VkDescriptorSet> descriptorSets = {DescriptorSets[frameNumber], object->getDescriptorSet()[frameNumber]};

            PushConstBlock BB;

            uint32_t primitiveCount = 0;

            object->getModel()->renderBB(
                object->getInstanceNumber(frameNumber),
                commandBuffers,
                PipelineLayout,
                static_cast<uint32_t>(descriptorSets.size()),
                descriptorSets.data(),
                primitiveCount,
                sizeof(PushConstBlock),
                0,
                &BB);
        }
    }
}
