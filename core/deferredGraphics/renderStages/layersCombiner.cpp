#include "layersCombiner.h"
#include "operations.h"
#include "vkdefault.h"
#include "texture.h"

layersCombiner::layersCombiner(bool enable, uint32_t transparentLayersCount, bool enableScatteringRefraction) :
    enable(enable)
{
    combiner.transparentLayersCount = transparentLayersCount == 0 ? 1 : transparentLayersCount;
    combiner.enableTransparentLayers = transparentLayersCount != 0;
    combiner.enableScatteringRefraction = enableScatteringRefraction;
}

void layersCombiner::setTransparentLayersCount(uint32_t transparentLayersCount){
    combiner.transparentLayersCount = transparentLayersCount;
}

void layersCombiner::setScatteringRefraction(bool enable){
    combiner.enableScatteringRefraction = enable;
}

void layersCombiner::setBlurDepth(float blurDepth){
    this->blurDepth = blurDepth;
}

void layersCombiner::createAttachments(attachmentsDatabase& aDatabase)
{
    auto createAttachments = [](VkPhysicalDevice physicalDevice, VkDevice device, const imageInfo image, uint32_t attachmentsCount, attachments* pAttachments){
        for(size_t index=0; index < attachmentsCount; index++){
            pAttachments[index].create(physicalDevice,device,image.Format,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | (index==1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0),image.Extent,image.Count);
            VkSamplerCreateInfo samplerInfo = vkDefault::samler();
            CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &pAttachments[index].sampler));
        }
    };

    createAttachments(physicalDevice, device, image, layersCombinerAttachments::size(), &frame);
    aDatabase.addAttachmentData("combined.color", enable, &frame.color);
    aDatabase.addAttachmentData("combined.bloom", enable, &frame.bloom);
    aDatabase.addAttachmentData("combined.blur", enable, &frame.blur);
}

void layersCombiner::destroy(){
    combiner.destroy(device);
    workflow::destroy();

    frame.deleteAttachment(device);
    frame.deleteSampler(device);
}

void layersCombiner::createRenderPass(){
    std::vector<VkAttachmentDescription> attachments = {
        attachments::imageDescription(image.Format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        attachments::imageDescription(image.Format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        attachments::imageDescription(image.Format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    };

    std::vector<std::vector<VkAttachmentReference>> attachmentRef;
    attachmentRef.push_back(std::vector<VkAttachmentReference>());
        attachmentRef.back().push_back(VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        attachmentRef.back().push_back(VkAttachmentReference{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        attachmentRef.back().push_back(VkAttachmentReference{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

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

void layersCombiner::createFramebuffers(){
    framebuffers.resize(image.Count);
    for(size_t i = 0; i < image.Count; i++){
        std::vector<VkImageView> attachments = {
            frame.color.instances[i].imageView,
            frame.bloom.instances[i].imageView,
            frame.blur.instances[i].imageView,
        };
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[i]));
    }
}

void layersCombiner::createPipelines(){
    combiner.vertShaderPath = shadersPath / "layersCombiner/layersCombinerVert.spv";
    combiner.fragShaderPath = shadersPath / "layersCombiner/layersCombinerFrag.spv";
    combiner.createDescriptorSetLayout(device);
    combiner.createPipeline(device,&image,renderPass);
}

void layersCombiner::Combiner::createDescriptorSetLayout(VkDevice device){
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(vkDefault::bufferFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), transparentLayersCount));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), transparentLayersCount));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), transparentLayersCount));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), transparentLayersCount));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), transparentLayersCount));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &DescriptorSetLayout));
}

void layersCombiner::Combiner::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass){
    uint32_t specializationData = transparentLayersCount;
    VkSpecializationMapEntry specializationMapEntry{};
        specializationMapEntry.constantID = 0;
        specializationMapEntry.offset = 0;
        specializationMapEntry.size = sizeof(uint32_t);
    VkSpecializationInfo specializationInfo;
        specializationInfo.mapEntryCount = 1;
        specializationInfo.pMapEntries = &specializationMapEntry;
        specializationInfo.dataSize = sizeof(specializationData);
        specializationInfo.pData = &specializationData;

    auto vertShaderCode = ShaderModule::readFile(vertShaderPath);
    auto fragShaderCode = ShaderModule::readFile(fragShaderPath);
    VkShaderModule vertShaderModule = ShaderModule::create(&device, vertShaderCode);
    VkShaderModule fragShaderModule = ShaderModule::create(&device, fragShaderCode);
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    shaderStages.push_back(VkPipelineShaderStageCreateInfo{});
        shaderStages.back().sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages.back().stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages.back().module = vertShaderModule;
        shaderStages.back().pName = "main";
    shaderStages.push_back(VkPipelineShaderStageCreateInfo{});
        shaderStages.back().sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages.back().stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages.back().module = fragShaderModule;
        shaderStages.back().pName = "main";
        shaderStages.back().pSpecializationInfo = &specializationInfo;

    VkViewport viewport = vkDefault::viewport({0,0}, pInfo->Extent);
    VkRect2D scissor = vkDefault::scissor({0,0}, pInfo->Extent);
    VkPipelineViewportStateCreateInfo viewportState = vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = vkDefault::depthStencilDisable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment(layersCombinerAttachments::size(), vkDefault::colorBlendAttachmentState(VK_FALSE));
    VkPipelineColorBlendStateCreateInfo colorBlending = vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(layersCombinerPushConst);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &DescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &PipelineLayout));

    VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext = nullptr;
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = PipelineLayout;
        pipelineInfo.renderPass = pRenderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.pDepthStencilState = &depthStencil;
    CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Pipeline));

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void layersCombiner::createDescriptorPool(){
    workflow::createDescriptorPool(device, &combiner, image.Count, (9 + 5 * combiner.transparentLayersCount) * image.Count, combiner.transparentLayersCount * image.Count);
}

void layersCombiner::createDescriptorSets(){
    workflow::createDescriptorSets(device, &combiner, image.Count);
}

void layersCombiner::create(attachmentsDatabase& aDatabase)
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

void layersCombiner::updateDescriptorSets(
    const std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>>& bufferMap,
    const attachmentsDatabase& aDatabase)
{
    if(!enable) return;

    for (uint32_t i = 0; i < image.Count; i++)
    {
        VkDescriptorBufferInfo bufferInfo;
            bufferInfo.buffer = bufferMap.at("camera").second[i];
            bufferInfo.offset = 0;
            bufferInfo.range = bufferMap.at("camera").first;

        VkDescriptorImageInfo colorImageInfo = aDatabase.descriptorImageInfo("image", i);
        VkDescriptorImageInfo bloomImageInfo = aDatabase.descriptorImageInfo("bloom", i);
        VkDescriptorImageInfo positionImageInfo = aDatabase.descriptorImageInfo("GBuffer.position", i);
        VkDescriptorImageInfo normalImageInfo = aDatabase.descriptorImageInfo("GBuffer.normal", i);
        VkDescriptorImageInfo depthImageInfo = aDatabase.descriptorImageInfo("GBuffer.depth", i, "white");
        VkDescriptorImageInfo skyboxImageInfo = aDatabase.descriptorImageInfo("skybox.color", i);
        VkDescriptorImageInfo skyboxBloomImageInfo = aDatabase.descriptorImageInfo("skybox.bloom", i);
        VkDescriptorImageInfo scatteringImageInfo = aDatabase.descriptorImageInfo("scattering", i);
        VkDescriptorImageInfo sslrImageInfo = aDatabase.descriptorImageInfo("sslr", i);

        std::vector<VkDescriptorImageInfo> colorLayersImageInfo(combiner.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> bloomLayersImageInfo(combiner.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> positionLayersImageInfo(combiner.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> normalLayersImageInfo(combiner.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> depthLayersImageInfo(combiner.transparentLayersCount);

        for(uint32_t index = 0; index < combiner.transparentLayersCount; index++){
            std::string key = "transparency" + std::to_string(index) + ".";

            colorLayersImageInfo[index] = aDatabase.descriptorImageInfo(key + "image", i);
            bloomLayersImageInfo[index] = aDatabase.descriptorImageInfo(key + "bloom", i);
            positionLayersImageInfo[index] = aDatabase.descriptorImageInfo(key + "GBuffer.position", i);
            normalLayersImageInfo[index] = aDatabase.descriptorImageInfo(key + "GBuffer.normal", i);
            depthLayersImageInfo[index] = aDatabase.descriptorImageInfo(key + "GBuffer.depth", i, "white");
        }

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(static_cast<uint32_t>(descriptorWrites.size() - 1));
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(static_cast<uint32_t>(descriptorWrites.size() - 1));
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &colorImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &bloomImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &positionImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &normalImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &depthImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = combiner.transparentLayersCount;
            descriptorWrites.back().pImageInfo = colorLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = combiner.transparentLayersCount;
            descriptorWrites.back().pImageInfo = bloomLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = combiner.transparentLayersCount;
            descriptorWrites.back().pImageInfo = positionLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = combiner.transparentLayersCount;
            descriptorWrites.back().pImageInfo = normalLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = combiner.transparentLayersCount;
            descriptorWrites.back().pImageInfo = depthLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &skyboxImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &skyboxBloomImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &scatteringImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &sslrImageInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void layersCombiner::updateCommandBuffer(uint32_t frameNumber){
    if(!enable) return;

    std::vector<VkClearValue> clearValues = {frame.color.clearValue, frame.bloom.clearValue, frame.blur.clearValue};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        layersCombinerPushConst pushConst{};
            pushConst.enableScatteringRefraction = static_cast<int>(combiner.enableScatteringRefraction);
            pushConst.enableTransparentLayers = static_cast<int>(combiner.enableTransparentLayers);
            pushConst.blurDepth = blurDepth;
        vkCmdPushConstants(commandBuffers[frameNumber], combiner.PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(layersCombinerPushConst), &pushConst);

        vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, combiner.Pipeline);
        vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, combiner.PipelineLayout, 0, 1, &combiner.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers[frameNumber], 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}
