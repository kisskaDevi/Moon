#include "boundingBoxGraphics.h"

#include "operations.h"
#include "vkdefault.h"
#include "matrix.h"

#include <cstring>

struct CameraBuffer{
    alignas(16) matrix<float,4,4> proj;
    alignas(16) matrix<float,4,4> view;
};

boundingBoxGraphics::boundingBoxGraphics() {}

void boundingBoxGraphics::createAttachments() {
    ::createAttachments(physicalDevice, device, image, 1, &frame);
}

void boundingBoxGraphics::boundingBoxGraphics::createRenderPass(){
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

void boundingBoxGraphics::createDescriptorSetLayout(){
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));
}

void boundingBoxGraphics::createPipeline(){
    auto vertShaderCode = ShaderModule::readFile(vertShaderPath);
    auto fragShaderCode = ShaderModule::readFile(fragShaderPath);
    VkShaderModule vertShaderModule = ShaderModule::create(&device, vertShaderCode);
    VkShaderModule fragShaderModule = ShaderModule::create(&device, fragShaderCode);
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        vkDefault::vertrxShaderStage(vertShaderModule),
        vkDefault::fragmentShaderStage(fragShaderModule)
    };

    VkViewport viewport = vkDefault::viewport({0,0}, image.Extent);
    VkRect2D scissor = vkDefault::scissor({0,0}, image.Extent);
    VkPipelineViewportStateCreateInfo viewportState = vkDefault::viewportState(&viewport, &scissor);
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = vkDefault::inputAssembly();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = vkDefault::vertexInputState();
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
    pushConstantRange.back().size = sizeof(cuda::cbox);

    std::vector<VkDescriptorSetLayout> setLayouts = {descriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pipelineLayoutInfo.pSetLayouts = setLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
    pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

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
    pipelineInfo.back().renderPass = renderPass;
    pipelineInfo.back().subpass = 0;
    pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &pipeline));

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
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
}

void boundingBoxGraphics::createDescriptorSets(){
    descriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
    allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data());

    for (uint32_t i = 0; i < image.Count; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = cameraBuffer[i].instance;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(CameraBuffer);

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
        descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites.back().dstSet = descriptorSets[i];
        descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
        descriptorWrites.back().dstArrayElement = 0;
        descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites.back().descriptorCount = 1;
        descriptorWrites.back().pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}


boundingBoxGraphics::~boundingBoxGraphics(){
    destroy();
}

void boundingBoxGraphics::destroy(){
    if(pipeline)            {vkDestroyPipeline(device, pipeline, nullptr); pipeline = VK_NULL_HANDLE;}
    if(pipelineLayout)      {vkDestroyPipelineLayout(device, pipelineLayout,nullptr); pipelineLayout = VK_NULL_HANDLE;}
    if(descriptorSetLayout) {vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr); descriptorSetLayout = VK_NULL_HANDLE;}
    if(descriptorPool)      {vkDestroyDescriptorPool(device, descriptorPool, nullptr); descriptorPool = VK_NULL_HANDLE;}

    if(renderPass) {vkDestroyRenderPass(device, renderPass, nullptr); renderPass = VK_NULL_HANDLE;}
    for(auto& framebuffer: framebuffers){
        if(framebuffer) vkDestroyFramebuffer(device, framebuffer,nullptr);
    }
    framebuffers.clear();

    frame.deleteAttachment(device);
    frame.deleteSampler(device);

    for (auto& buffer: cameraBuffer){
        Buffer::destroy(device, buffer.instance, buffer.memory);
    }
    cameraBuffer.clear();
}

void boundingBoxGraphics::create(VkPhysicalDevice physicalDevice, VkDevice device, const imageInfo& image, const std::filesystem::path& shadersPath){
    if(!enable) return;

    this->physicalDevice = physicalDevice;
    this->device = device;
    this->image = image;
    vertShaderPath = shadersPath / "boundingBox/boundingBoxVert.spv";
    fragShaderPath = shadersPath / "boundingBox/boundingBoxFrag.spv";

    cameraBuffer.resize(image.Count);
    for (auto& buffer: cameraBuffer){
        Buffer::create(physicalDevice, device, sizeof(CameraBuffer), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer.instance, &buffer.memory);
    }

    createAttachments();
    createRenderPass();
    createFramebuffers();
    createDescriptorSetLayout();
    createPipeline();
    createDescriptorPool();
    createDescriptorSets();
}

void boundingBoxGraphics::update(uint32_t imageIndex){
    if(!enable) return;

    cuda::camera hostCam = cuda::to_host(*camera);
    const float fov = 2.0f * std::atan(hostCam.matrixScale / hostCam.matrixOffset);
    const auto& u =  normal(hostCam.horizontal);
    const auto& v =  normal(hostCam.vertical);
    const auto& n = -normal(hostCam.viewRay.getDirection());
    const auto& c = hostCam.viewRay.getOrigin();

    matrix<float,4,4> projMatrix = perspective(fov, hostCam.aspect, hostCam.matrixOffset);
    matrix<float,4,4> viewMatrix = {
        u[0], u[1], u[2], - (c[0]*u[0] + c[1]*u[1] + c[2]*u[2]),
        v[0], v[1], v[2], - (c[0]*v[0] + c[1]*v[1] + c[2]*v[2]),
        n[0], n[1], n[2], - (c[0]*n[0] + c[1]*n[1] + c[2]*n[2]),
        0.0f, 0.0f, 0.0f, 1.0f
    };

    CameraBuffer buffer{};
    buffer.proj = transpose(projMatrix);
    buffer.view = transpose(viewMatrix);

    CHECK(vkMapMemory(device, cameraBuffer[imageIndex].memory, 0, sizeof(CameraBuffer), 0, &cameraBuffer[imageIndex].map));
    std::memcpy(cameraBuffer[imageIndex].map, &buffer, sizeof(CameraBuffer));
    vkUnmapMemory(device, cameraBuffer[imageIndex].memory);
}

void boundingBoxGraphics::render(VkCommandBuffer commandBuffer, uint32_t imageIndex) const {
    if(!enable) return;

    std::vector<VkClearValue> clearValues = {frame.clearValue};

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = framebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0,0};
    renderPassInfo.renderArea.extent = image.Extent;
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[imageIndex], 0, NULL);
    for(auto box: boxes){
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(cuda::cbox), &box);
        vkCmdDraw(commandBuffer, 36, 1, 0, 0);
    }

    vkCmdEndRenderPass(commandBuffer);
}

const attachments& boundingBoxGraphics::getAttachments() const {
    return frame;
}

void boundingBoxGraphics::bind(cuda::cbox box){
    boxes.push_back(box);
}

void boundingBoxGraphics::bind(cuda::devicep<cuda::camera>* camera){
    this->camera = camera;
}
