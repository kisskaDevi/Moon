#include "../graphics.h"
#include "operations.h"
#include "vkdefault.h"
#include "texture.h"
#include "model.h"
#include "object.h"

namespace moon::deferredGraphics {

Graphics::Base::Base(
    const bool transparencyPass,
    const bool enableTransparency,
    const uint32_t transparencyNumber,
    const utils::ImageInfo& imageInfo,
    const GraphicsParameters& parameters) :
    transparencyPass(transparencyPass),
    enableTransparency(enableTransparency),
    transparencyNumber(transparencyNumber),
    imageInfo(imageInfo),
    parameters(parameters)
{}

void Graphics::Base::createDescriptorSetLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(moon::utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    CHECK(descriptorSetLayout.create(device, bindings));

    objectDescriptorSetLayout = moon::interfaces::Object::createDescriptorSetLayout(device);
    primitiveDescriptorSetLayout = moon::interfaces::Model::createNodeDescriptorSetLayout(device);
    materialDescriptorSetLayout = moon::interfaces::Model::createMaterialDescriptorSetLayout(device);
}

void Graphics::Base::createPipeline(VkRenderPass pRenderPass) {
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

    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, shadersPath / "base/baseVert.spv");
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, shadersPath / "base/baseFrag.spv", specializationInfo);
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {vertShader, fragShader};

    auto bindingDescription = moon::interfaces::Model::Vertex::getBindingDescription();
    auto attributeDescriptions = moon::interfaces::Model::Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkViewport viewport = moon::utils::vkDefault::viewport({0,0}, imageInfo.Extent);
    VkRect2D scissor = moon::utils::vkDefault::scissor({0,0}, imageInfo.Extent);
    VkPipelineViewportStateCreateInfo viewportState = moon::utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = moon::utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = moon::utils::vkDefault::rasterizationState(VK_FRONT_FACE_COUNTER_CLOCKWISE);
    VkPipelineMultisampleStateCreateInfo multisampling = moon::utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = moon::utils::vkDefault::depthStencilEnable();

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
        pushConstantRange.back().size = sizeof(moon::interfaces::MaterialBlock);
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
        descriptorSetLayout,
        objectDescriptorSetLayout,
        primitiveDescriptorSetLayout,
        materialDescriptorSetLayout
    };
    CHECK(pipelineLayoutMap[moon::interfaces::ObjectType::base].create(device, descriptorSetLayouts, pushConstantRange));

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
        pipelineInfo.back().layout = pipelineLayoutMap[moon::interfaces::ObjectType::base];
        pipelineInfo.back().renderPass = pRenderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    CHECK(pipelineMap[moon::interfaces::ObjectType::base].create(device, pipelineInfo));

    moon::utils::vkDefault::MaskType outliningMask = moon::interfaces::ObjectType::base | moon::interfaces::ObjectProperty::outlining;
        depthStencil.stencilTestEnable = VK_TRUE;
        depthStencil.back.compareOp = VK_COMPARE_OP_ALWAYS;
        depthStencil.back.failOp = VK_STENCIL_OP_REPLACE;
        depthStencil.back.depthFailOp = VK_STENCIL_OP_REPLACE;
        depthStencil.back.passOp = VK_STENCIL_OP_REPLACE;
        depthStencil.back.compareMask = 0xff;
        depthStencil.back.writeMask = 0xff;
        depthStencil.back.reference = 1;
        depthStencil.front = depthStencil.back;
    CHECK(pipelineLayoutMap[outliningMask].create(device, descriptorSetLayouts, pushConstantRange));
        pipelineInfo.back().layout = pipelineLayoutMap[outliningMask];
    CHECK(pipelineMap[outliningMask].create(device, pipelineInfo));
}

void Graphics::Base::createDescriptorPool() {
    CHECK(descriptorPool.create(device, { &descriptorSetLayout }, imageInfo.Count));
}

void Graphics::Base::createDescriptorSets()
{
    descriptorSets.resize(imageInfo.Count);
    std::vector<VkDescriptorSetLayout> layouts(imageInfo.Count, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageInfo.Count);
        allocInfo.pSetLayouts = layouts.data();
    CHECK(vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()));
}

void Graphics::Base::updateDescriptorSets(
    const moon::utils::BuffersDatabase& bDatabase,
    const moon::utils::AttachmentsDatabase& aDatabase)
{
    CHECK_M(device == VK_NULL_HANDLE, std::string("[ Graphics::Base::updateDescriptorSets ] VkDevice is VK_NULL_HANDLE"));

    for (uint32_t i = 0; i < imageInfo.Count; i++)
    {
        VkDescriptorBufferInfo bufferInfo = bDatabase.descriptorBufferInfo(parameters.in.camera, i);

        VkDescriptorImageInfo skyboxImageInfo{};
            skyboxImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            skyboxImageInfo.imageView = *aDatabase.getEmpty()->getTextureImageView();
            skyboxImageInfo.sampler = *aDatabase.getEmpty()->getTextureSampler();

        std::string depthId = !transparencyPass || transparencyNumber == 0 ? "" :
                                      (parameters.out.transparency + std::to_string(transparencyNumber - 1) + ".") + parameters.out.depth;
        VkDescriptorImageInfo depthImageInfo = aDatabase.descriptorImageInfo(depthId, i);

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &skyboxImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptorSets.at(i);
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &depthImageInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void Graphics::Base::create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass) {
    this->device = device;
    this->shadersPath = shadersPath;

    createDescriptorSetLayout();
    createPipeline(pRenderPass);
    createDescriptorPool();
    createDescriptorSets();
}

void Graphics::Base::render(uint32_t frameNumber, VkCommandBuffer commandBuffers, uint32_t& primitiveCount) const
{
    for(const auto& object: *objects){
        if(VkDeviceSize offsets = 0; (moon::interfaces::ObjectType::base & object->getPipelineBitMask()) && object->getEnable()){
            vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineMap.at(object->getPipelineBitMask()));

            vkCmdBindVertexBuffers(commandBuffers, 0, 1, object->getModel()->getVertices(), &offsets);
            if (object->getModel()->getIndices() != VK_NULL_HANDLE){
                vkCmdBindIndexBuffer(commandBuffers, *object->getModel()->getIndices(), 0, VK_INDEX_TYPE_UINT32);
            }

            std::vector<VkDescriptorSet> descriptors = { descriptorSets[frameNumber], object->getDescriptorSet(frameNumber)};

            moon::interfaces::MaterialBlock material;

            object->setFirstPrimitive(primitiveCount);
            object->getModel()->render(
                        object->getInstanceNumber(frameNumber),
                        commandBuffers,
                        pipelineLayoutMap.at(object->getPipelineBitMask()),
                        static_cast<uint32_t>(descriptors.size()),
                        descriptors.data(),
                        primitiveCount,
                        sizeof(moon::interfaces::MaterialBlock),
                        0,
                        &material);
            object->setPrimitiveCount(primitiveCount - object->getFirstPrimitive());
        }
    }
}

}
