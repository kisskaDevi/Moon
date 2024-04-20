#include "../graphics.h"
#include "operations.h"
#include "vkdefault.h"
#include "texture.h"
#include "model.h"
#include "object.h"

void graphics::Base::Destroy(VkDevice device)
{
    for(auto& PipelineLayout: PipelineLayoutDictionary){
        if(PipelineLayout.second) {vkDestroyPipelineLayout(device, PipelineLayout.second, nullptr); PipelineLayout.second = VK_NULL_HANDLE;}
    }
    for(auto& Pipeline: PipelineDictionary){
        if(Pipeline.second) {vkDestroyPipeline(device, Pipeline.second, nullptr); Pipeline.second = VK_NULL_HANDLE;}
    }
    if(SceneDescriptorSetLayout)        {vkDestroyDescriptorSetLayout(device, SceneDescriptorSetLayout,  nullptr); SceneDescriptorSetLayout = VK_NULL_HANDLE;}
    if(ObjectDescriptorSetLayout)       {vkDestroyDescriptorSetLayout(device, ObjectDescriptorSetLayout,  nullptr); ObjectDescriptorSetLayout = VK_NULL_HANDLE;}
    if(PrimitiveDescriptorSetLayout)    {vkDestroyDescriptorSetLayout(device, PrimitiveDescriptorSetLayout,  nullptr); PrimitiveDescriptorSetLayout = VK_NULL_HANDLE;}
    if(MaterialDescriptorSetLayout)     {vkDestroyDescriptorSetLayout(device, MaterialDescriptorSetLayout,  nullptr); MaterialDescriptorSetLayout = VK_NULL_HANDLE;}
    if(DescriptorPool)                  {vkDestroyDescriptorPool(device, DescriptorPool, nullptr); DescriptorPool = VK_NULL_HANDLE;}
}

void graphics::Base::createDescriptorSetLayout(VkDevice device)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &SceneDescriptorSetLayout));

    object::createDescriptorSetLayout(device,&ObjectDescriptorSetLayout);
    model::createNodeDescriptorSetLayout(device,&PrimitiveDescriptorSetLayout);
    model::createMaterialDescriptorSetLayout(device,&MaterialDescriptorSetLayout);
}

void graphics::Base::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass)
{
    std::vector<VkBool32> transparencyData = {
        static_cast<VkBool32>(enableTransparency),
        static_cast<VkBool32>(transparencyPass)
    };
    std::vector<VkSpecializationMapEntry> specializationMapEntry;
    specializationMapEntry.push_back(VkSpecializationMapEntry{});
        specializationMapEntry.back().constantID = static_cast<uint32_t>(specializationMapEntry.size() - 1);
        specializationMapEntry.back().offset = 0;
        specializationMapEntry.back().size = sizeof(VkBool32);
    specializationMapEntry.push_back(VkSpecializationMapEntry{});
        specializationMapEntry.back().constantID = static_cast<uint32_t>(specializationMapEntry.size() - 1);
        specializationMapEntry.back().offset = sizeof(VkBool32);
        specializationMapEntry.back().size = sizeof(VkBool32);
    VkSpecializationInfo specializationInfo;
        specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntry.size());
        specializationInfo.pMapEntries = specializationMapEntry.data();
        specializationInfo.dataSize = sizeof(VkBool32) * transparencyData.size();
        specializationInfo.pData = transparencyData.data();

    auto vertShaderCode = ShaderModule::readFile(ShadersPath / "base/baseVert.spv");
    auto fragShaderCode = ShaderModule::readFile(ShadersPath / "base/baseFrag.spv");
    VkShaderModule vertShaderModule = ShaderModule::create(&device, vertShaderCode);
    VkShaderModule fragShaderModule = ShaderModule::create(&device, fragShaderCode);
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
        vkDefault::vertrxShaderStage(vertShaderModule),
        vkDefault::fragmentShaderStage(fragShaderModule)
    };
    shaderStages.back().pSpecializationInfo = &specializationInfo;

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
    VkPipelineDepthStencilStateCreateInfo depthStencil = vkDefault::depthStencilEnable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {
        vkDefault::colorBlendAttachmentState(VK_FALSE),
        vkDefault::colorBlendAttachmentState(VK_FALSE),
        vkDefault::colorBlendAttachmentState(VK_FALSE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(MaterialBlock);
    std::vector<VkDescriptorSetLayout> setLayouts = {
        SceneDescriptorSetLayout,
        ObjectDescriptorSetLayout,
        PrimitiveDescriptorSetLayout,
        MaterialDescriptorSetLayout
    };
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
        pipelineLayoutInfo.pSetLayouts = setLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange.data();
    CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &PipelineLayoutDictionary[objectType::base]));

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
        pipelineInfo.back().layout = PipelineLayoutDictionary[objectType::base];
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &PipelineDictionary[objectType::base]));

        depthStencil.stencilTestEnable = VK_TRUE;
        depthStencil.back.compareOp = VK_COMPARE_OP_ALWAYS;
        depthStencil.back.failOp = VK_STENCIL_OP_REPLACE;
        depthStencil.back.depthFailOp = VK_STENCIL_OP_REPLACE;
        depthStencil.back.passOp = VK_STENCIL_OP_REPLACE;
        depthStencil.back.compareMask = 0xff;
        depthStencil.back.writeMask = 0xff;
        depthStencil.back.reference = 1;
        depthStencil.front = depthStencil.back;
    CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &PipelineLayoutDictionary[objectType::base | objectProperty::outlining]));
        pipelineInfo.back().layout = PipelineLayoutDictionary[objectType::base | objectProperty::outlining];
    CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &PipelineDictionary[objectType::base | objectProperty::outlining]));

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void graphics::createBaseDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(2 * image.Count);
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &base.DescriptorPool));
}

void graphics::createBaseDescriptorSets()
{
    base.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, base.SceneDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = base.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();
    CHECK(vkAllocateDescriptorSets(device, &allocInfo, base.DescriptorSets.data()));
}

void graphics::updateBaseDescriptorSets(
    const buffersDatabase& bDatabase,
    const attachmentsDatabase& aDatabase)
{
    for (uint32_t i = 0; i < image.Count; i++)
    {
        VkDescriptorBufferInfo bufferInfo = bDatabase.descriptorBufferInfo("camera", i);

        VkDescriptorImageInfo skyboxImageInfo{};
            skyboxImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            skyboxImageInfo.imageView = *emptyTexture["black"]->getTextureImageView();
            skyboxImageInfo.sampler   = *emptyTexture["black"]->getTextureSampler();

        std::string depthId = !base.transparencyPass || base.transparencyNumber == 0 ? "" : ("transparency" + std::to_string(base.transparencyNumber - 1) + ".") + "GBuffer.depth";
        VkDescriptorImageInfo depthImageInfo = aDatabase.descriptorImageInfo(depthId, i);

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = base.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = base.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &skyboxImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = base.DescriptorSets.at(i);
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &depthImageInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void graphics::Base::render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount)
{
    for(auto object: *objects){
        if(VkDeviceSize offsets = 0; (objectType::base & object->getPipelineBitMask()) && object->getEnable()){
            vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineDictionary[object->getPipelineBitMask()]);

            vkCmdBindVertexBuffers(commandBuffers, 0, 1, object->getModel()->getVertices(), &offsets);
            if (object->getModel()->getIndices() != VK_NULL_HANDLE){
                vkCmdBindIndexBuffer(commandBuffers, *object->getModel()->getIndices(), 0, VK_INDEX_TYPE_UINT32);
            }

            std::vector<VkDescriptorSet> descriptorSets = {
                DescriptorSets[frameNumber],
                object->getDescriptorSet()[frameNumber]
            };

            MaterialBlock material;

            object->setFirstPrimitive(primitiveCount);
            object->getModel()->render(
                        object->getInstanceNumber(frameNumber),
                        commandBuffers,
                        PipelineLayoutDictionary[object->getPipelineBitMask()],
                        static_cast<uint32_t>(descriptorSets.size()),
                        descriptorSets.data(),
                        primitiveCount,
                        sizeof(MaterialBlock),
                        0,
                        &material);
            object->setPrimitiveCount(primitiveCount - object->getFirstPrimitive());
        }
    }
}
