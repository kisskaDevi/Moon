#include "../graphics.h"
#include "../../../utils/operations.h"
#include "../../../utils/vkdefault.h"
#include "../../../utils/texture.h"
#include "../../../interfaces/model.h"
#include "../../../interfaces/object.h"
#include "../../../interfaces/camera.h"

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
    bindings.push_back(VkDescriptorSetLayoutBinding{});
       bindings.back().binding = static_cast<uint32_t>(bindings.size() - 1);
       bindings.back().descriptorCount = 1;
       bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
       bindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
       bindings.back().pImmutableSamplers = nullptr;
    bindings.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &SceneDescriptorSetLayout);

    object::createDescriptorSetLayout(device,&ObjectDescriptorSetLayout);
    model::createNodeDescriptorSetLayout(device,&PrimitiveDescriptorSetLayout);
    model::createMaterialDescriptorSetLayout(device,&MaterialDescriptorSetLayout);
}

void graphics::Base::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass)
{
    auto vertShaderCode = ShaderModule::readFile(ExternalPath + "core\\deferredGraphics\\shaders\\base\\basevert.spv");
    auto fragShaderCode = ShaderModule::readFile(ExternalPath + "core\\deferredGraphics\\shaders\\base\\basefrag.spv");
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
    VkPipelineDepthStencilStateCreateInfo depthStencil = vkDefault::depthStencilEnable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {
        vkDefault::colorBlendAttachmentState(VK_FALSE),
        vkDefault::colorBlendAttachmentState(VK_FALSE),
        vkDefault::colorBlendAttachmentState(VK_FALSE),
        vkDefault::colorBlendAttachmentState(VK_FALSE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    struct PushConstBlock {float transparencyPass; MaterialBlock material;};
    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(PushConstBlock);
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
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &PipelineLayoutDictionary[(0<<4)|0]);

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
        pipelineInfo.back().layout = PipelineLayoutDictionary[(0<<4)|0];
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &PipelineDictionary[(0<<4)|0]);

    depthStencil = vkDefault::depthStencilEnable();
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &PipelineLayoutDictionary[(1<<4)|0]);
    pipelineInfo.back().layout = PipelineLayoutDictionary[(1<<4)|0];
    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, static_cast<uint32_t>(pipelineInfo.size()), pipelineInfo.data(), nullptr, &PipelineDictionary[(1<<4)|0]);

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
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(image.Count);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(2 * image.Count);
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &base.DescriptorPool);
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
    vkAllocateDescriptorSets(device, &allocInfo, base.DescriptorSets.data());
}

void graphics::updateBaseDescriptorSets(attachments* depthAttachment, VkBuffer* storageBuffers, size_t sizeOfStorageBuffers, camera* cameraObject)
{
    for (size_t i = 0; i < image.Count; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = cameraObject->getBuffer(i);
            bufferInfo.offset = 0;
            bufferInfo.range = cameraObject->getBufferRange();

        VkDescriptorImageInfo skyboxImageInfo{};
            skyboxImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            skyboxImageInfo.imageView = *emptyTexture->getTextureImageView();
            skyboxImageInfo.sampler   = *emptyTexture->getTextureSampler();

        VkDescriptorBufferInfo StorageBufferInfo{};
            StorageBufferInfo.buffer = storageBuffers[i];
            StorageBufferInfo.offset = 0;
            StorageBufferInfo.range = sizeOfStorageBuffers;

        VkDescriptorImageInfo depthImageInfo{};
            depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            depthImageInfo.imageView = depthAttachment ? depthAttachment->imageView[i] : *emptyTexture->getTextureImageView();
            depthImageInfo.sampler = depthAttachment ? depthAttachment->sampler : *emptyTexture->getTextureSampler();

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
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &StorageBufferInfo;
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
    for(auto object: objects){
        if(VkDeviceSize offsets = 0; object->getEnable()){
            vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineDictionary[object->getPipelineBitMask()]);

            vkCmdBindVertexBuffers(commandBuffers, 0, 1, object->getModel()->getVertices(), &offsets);
            if (object->getModel()->getIndices() != VK_NULL_HANDLE){
                vkCmdBindIndexBuffer(commandBuffers, *object->getModel()->getIndices(), 0, VK_INDEX_TYPE_UINT32);
            }

            object->resetPrimitiveCount();
            object->setFirstPrimitive(primitiveCount);

            std::vector<VkDescriptorSet> descriptorSets = {DescriptorSets[frameNumber],object->getDescriptorSet()[frameNumber]};

            struct PushConstBlock{
                float transparencyPass;
                MaterialBlock material;
            } pushConstBlock;
            pushConstBlock.transparencyPass = transparencyPass ? 1.0f : 0.0f;

            object->getModel()->render(
                        object->getInstanceNumber(frameNumber),
                        commandBuffers,
                        PipelineLayoutDictionary[object->getPipelineBitMask()],
                        static_cast<uint32_t>(descriptorSets.size()),
                        descriptorSets.data(),
                        primitiveCount,
                        sizeof(PushConstBlock),
                        offsetof(PushConstBlock, material),
                        &pushConstBlock);
            object->setPrimitiveCount(primitiveCount - object->getFirstPrimitive());
        }
    }
}
