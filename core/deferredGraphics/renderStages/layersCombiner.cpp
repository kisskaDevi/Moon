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

void layersCombiner::createAttachments(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap)
{
    auto createAttachments = [](VkPhysicalDevice physicalDevice, VkDevice device, const imageInfo image, uint32_t attachmentsCount, attachments* pAttachments){
        for(size_t index=0; index < attachmentsCount; index++){
            pAttachments[index].create(physicalDevice,device,image.Format,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | (index==1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0),image.frameBufferExtent,image.Count);
            VkSamplerCreateInfo samplerInfo = vkDefault::samler();
            vkCreateSampler(device, &samplerInfo, nullptr, &pAttachments[index].sampler);
        }
    };

    createAttachments(physicalDevice, device, image, 2, &frame);
    attachmentsMap["combined.color"] = {enable, {&frame.color}};
    attachmentsMap["combined.bloom"] = {enable, {&frame.bloom}};
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
        attachments::imageDescription(image.Format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    };

    std::vector<std::vector<VkAttachmentReference>> attachmentRef;
    attachmentRef.push_back(std::vector<VkAttachmentReference>());
        attachmentRef.back().push_back(VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        attachmentRef.back().push_back(VkAttachmentReference{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

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
    vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass);
}

void layersCombiner::createFramebuffers(){
    framebuffers.resize(image.Count);
    for(size_t i = 0; i < image.Count; i++){
        std::vector<VkImageView> attachments = {
            frame.color.instances[i].imageView,
            frame.bloom.instances[i].imageView
        };
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = image.frameBufferExtent.width;
            framebufferInfo.height = image.frameBufferExtent.height;
            framebufferInfo.layers = 1;
        vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[i]);
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

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &DescriptorSetLayout);
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

    VkViewport viewport = vkDefault::viewport(pInfo->Offset, pInfo->Extent);
    VkRect2D scissor = vkDefault::scissor({0,0}, pInfo->frameBufferExtent);
    VkPipelineViewportStateCreateInfo viewportState = vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = vkDefault::depthStencilDisable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment(2,vkDefault::colorBlendAttachmentState(VK_FALSE));
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
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &PipelineLayout);

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
    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Pipeline);

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void layersCombiner::createDescriptorPool(){
    workflow::createDescriptorPool(device, &combiner, image.Count, (8 + 5 * combiner.transparentLayersCount) * image.Count, combiner.transparentLayersCount * image.Count);
}

void layersCombiner::createDescriptorSets(){
    workflow::createDescriptorSets(device, &combiner, image.Count);
}

void layersCombiner::create(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap)
{
    if(enable){
        createAttachments(attachmentsMap);
        createRenderPass();
        createFramebuffers();
        createPipelines();
        createDescriptorPool();
        createDescriptorSets();
    }
}

void layersCombiner::updateDescriptorSets(
    const std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>>& bufferMap,
    const std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap)
{
    for (uint32_t i = 0; i < image.Count; i++)
    {
        VkDescriptorBufferInfo bufferInfo;
            bufferInfo.buffer = bufferMap.at("camera").second[i];
            bufferInfo.offset = 0;
            bufferInfo.range = bufferMap.at("camera").first;

        const auto deferredAttachmentsImage = attachmentsMap.at("image").second.front();
        VkDescriptorImageInfo colorImageInfo;
            colorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            colorImageInfo.imageView = deferredAttachmentsImage->instances[i].imageView;
            colorImageInfo.sampler = deferredAttachmentsImage->sampler;

        const auto deferredAttachmentsBloom = attachmentsMap.at("bloom").second.front();
        VkDescriptorImageInfo bloomImageInfo;
            bloomImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            bloomImageInfo.imageView = deferredAttachmentsBloom->instances[i].imageView;
            bloomImageInfo.sampler = deferredAttachmentsBloom->sampler;

        const auto deferredAttachmentsGPos = attachmentsMap.at("GBuffer.position").second.front();
        VkDescriptorImageInfo positionImageInfo;
            positionImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            positionImageInfo.imageView = deferredAttachmentsGPos->instances[i].imageView;
            positionImageInfo.sampler = deferredAttachmentsGPos->sampler;

        const auto deferredAttachmentsGNorm = attachmentsMap.at("GBuffer.normal").second.front();
        VkDescriptorImageInfo normalImageInfo;
            normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            normalImageInfo.imageView = deferredAttachmentsGNorm->instances[i].imageView;
            normalImageInfo.sampler = deferredAttachmentsGNorm->sampler;

        const auto deferredAttachmentsGDepth = attachmentsMap.at("GBuffer.depth").second.front();
        VkDescriptorImageInfo depthImageInfo;
            depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            depthImageInfo.imageView = deferredAttachmentsGDepth->instances[i].imageView;
            depthImageInfo.sampler = deferredAttachmentsGDepth->sampler;

        const auto skyboxColor = attachmentsMap.count("skybox.color") > 0 && attachmentsMap.at("skybox.color").first ? attachmentsMap.at("skybox.color").second.front() : nullptr;
        VkDescriptorImageInfo skyboxImageInfo;
            skyboxImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            skyboxImageInfo.imageView = skyboxColor ? skyboxColor->instances[i].imageView : *emptyTexture["black"]->getTextureImageView();
            skyboxImageInfo.sampler = skyboxColor ? skyboxColor->sampler : *emptyTexture["black"]->getTextureSampler();

        const auto skyboxBloom = attachmentsMap.count("skybox.bloom") > 0 && attachmentsMap.at("skybox.bloom").first ? attachmentsMap.at("skybox.bloom").second.front() : nullptr;
        VkDescriptorImageInfo skyboxBloomImageInfo;
            skyboxBloomImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            skyboxBloomImageInfo.imageView = skyboxBloom ? skyboxBloom->instances[i].imageView : *emptyTexture["black"]->getTextureImageView();
            skyboxBloomImageInfo.sampler = skyboxBloom ? skyboxBloom->sampler : *emptyTexture["black"]->getTextureSampler();

        const auto scattering = attachmentsMap.count("scattering") > 0 && attachmentsMap.at("scattering").first ? attachmentsMap.at("scattering").second.front() : nullptr;
        VkDescriptorImageInfo scatteringImageInfo;
            scatteringImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            scatteringImageInfo.imageView = scattering ? scattering->instances[i].imageView : *emptyTexture["black"]->getTextureImageView();
            scatteringImageInfo.sampler = scattering ? scattering->sampler : *emptyTexture["black"]->getTextureSampler();

        std::vector<VkDescriptorImageInfo> colorLayersImageInfo(combiner.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> bloomLayersImageInfo(combiner.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> positionLayersImageInfo(combiner.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> normalLayersImageInfo(combiner.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> depthLayersImageInfo(combiner.transparentLayersCount);

        for(uint32_t index = 0; index < combiner.transparentLayersCount; index++){
            std::string key = "transparency" + std::to_string(index) + ".";

            const auto transparencyLayersImage = attachmentsMap.count(key + "image") > 0 && attachmentsMap.at(key + "image").first ? attachmentsMap.at(key + "image").second.front() : nullptr;
            colorLayersImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            colorLayersImageInfo[index].imageView = transparencyLayersImage ? transparencyLayersImage->instances[i].imageView : *emptyTexture["black"]->getTextureImageView();
            colorLayersImageInfo[index].sampler = transparencyLayersImage ? transparencyLayersImage->sampler : *emptyTexture["black"]->getTextureSampler();

            const auto transparencyLayersBloom = attachmentsMap.count(key + "bloom") > 0 && attachmentsMap.at(key + "bloom").first? attachmentsMap.at(key + "bloom").second.front() : nullptr;
            bloomLayersImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            bloomLayersImageInfo[index].imageView = transparencyLayersBloom ? transparencyLayersBloom->instances[i].imageView : *emptyTexture["black"]->getTextureImageView();
            bloomLayersImageInfo[index].sampler = transparencyLayersBloom ? transparencyLayersBloom->sampler : *emptyTexture["black"]->getTextureSampler();

            const auto transparencyLayersGPos = attachmentsMap.count(key + "GBuffer.position") > 0 && attachmentsMap.at(key + "GBuffer.position").first ? attachmentsMap.at(key + "GBuffer.position").second.front() : nullptr;
            positionLayersImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            positionLayersImageInfo[index].imageView = transparencyLayersGPos ? transparencyLayersGPos->instances[i].imageView : *emptyTexture["black"]->getTextureImageView();
            positionLayersImageInfo[index].sampler = transparencyLayersGPos ? transparencyLayersGPos->sampler : *emptyTexture["black"]->getTextureSampler();

            const auto transparencyLayersGNorm = attachmentsMap.count(key + "GBuffer.normal") > 0 && attachmentsMap.at(key + "GBuffer.normal").first ? attachmentsMap.at(key + "GBuffer.normal").second.front() : nullptr;
            normalLayersImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            normalLayersImageInfo[index].imageView = transparencyLayersGNorm ? transparencyLayersGNorm->instances[i].imageView : *emptyTexture["black"]->getTextureImageView();
            normalLayersImageInfo[index].sampler = transparencyLayersGNorm ? transparencyLayersGNorm->sampler : *emptyTexture["black"]->getTextureSampler();

            const auto transparencyLayersGDepth = attachmentsMap.count(key + "GBuffer.depth") > 0 && attachmentsMap.at(key + "GBuffer.depth").first ? attachmentsMap.at(key + "GBuffer.depth").second.front() : nullptr;
            depthLayersImageInfo[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            depthLayersImageInfo[index].imageView = transparencyLayersGDepth ? transparencyLayersGDepth->instances[i].imageView : *emptyTexture["white"]->getTextureImageView();
            depthLayersImageInfo[index].sampler = transparencyLayersGDepth ? transparencyLayersGDepth->sampler : *emptyTexture["white"]->getTextureSampler();
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
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void layersCombiner::updateCommandBuffer(uint32_t frameNumber)
{
    std::vector<VkClearValue> clearValues = {frame.color.clearValue, frame.bloom.clearValue};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = image.frameBufferExtent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        layersCombinerPushConst pushConst{};
            pushConst.enableScatteringRefraction = static_cast<int>(combiner.enableScatteringRefraction);
            pushConst.enableTransparentLayers = static_cast<int>(combiner.enableTransparentLayers);
        vkCmdPushConstants(commandBuffers[frameNumber], combiner.PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(layersCombinerPushConst), &pushConst);

        vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, combiner.Pipeline);
        vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, combiner.PipelineLayout, 0, 1, &combiner.DescriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers[frameNumber], 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}
