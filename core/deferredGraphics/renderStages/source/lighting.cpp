#include "graphics.h"
#include "vkdefault.h"
#include "light.h"
#include "deferredAttachments.h"
#include "depthMap.h"
#include "operations.h"

namespace moon::deferredGraphics {

Graphics::Lighting::Lighting(const utils::ImageInfo& imageInfo,
                             const GraphicsParameters& parameters) :
    enableScattering(enableScattering),
    imageInfo(imageInfo),
    parameters(parameters)
{}

void Graphics::Lighting::createDescriptorSetLayout()
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(moon::utils::vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(moon::utils::vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(moon::utils::vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(moon::utils::vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(moon::utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    CHECK(descriptorSetLayout.create(device, bindings));

    bufferDescriptorSetLayoutMap[moon::interfaces::LightType::spot] = moon::interfaces::Light::createBufferDescriptorSetLayout(device);
    textureDescriptorSetLayoutMap[moon::interfaces::LightType::spot] = moon::interfaces::Light::createTextureDescriptorSetLayout(device);
    shadowDescriptorSetLayout = moon::utils::DepthMap::createDescriptorSetLayout(device);
}

void Graphics::Lighting::createPipeline(VkRenderPass pRenderPass)
{
    std::filesystem::path spotVert = shadersPath / "spotLightingPass/spotLightingVert.spv";
    std::filesystem::path spotFrag = shadersPath / "spotLightingPass/spotLightingFrag.spv";
    createPipeline(moon::interfaces::LightType::spot, pRenderPass, spotVert, spotFrag);
}

void Graphics::Lighting::createDescriptors() {
    CHECK(descriptorPool.create(device, { &descriptorSetLayout }, imageInfo.Count));
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageInfo.Count);
}

void Graphics::Lighting::updateDescriptorSets(
    const moon::utils::BuffersDatabase& bDatabase,
    const moon::utils::AttachmentsDatabase& aDatabase)
{
    for (uint32_t i = 0; i < imageInfo.Count; i++){
        std::vector<VkDescriptorImageInfo> imageInfos;
        imageInfos.push_back(aDatabase.descriptorImageInfo(parameters.out.position, i));
        imageInfos.push_back(aDatabase.descriptorImageInfo(parameters.out.normal, i));
        imageInfos.push_back(aDatabase.descriptorImageInfo(parameters.out.color, i));
        imageInfos.push_back(aDatabase.descriptorImageInfo(parameters.out.depth, i));

        VkDescriptorBufferInfo bufferInfo = bDatabase.descriptorBufferInfo(parameters.in.camera, i);

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        for(auto& imageInfo: imageInfos){
            descriptorWrites.push_back(VkWriteDescriptorSet{});
                descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites.back().dstSet = descriptorSets[i];
                descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
                descriptorWrites.back().dstArrayElement = 0;
                descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
                descriptorWrites.back().descriptorCount = 1;
                descriptorWrites.back().pImageInfo = &imageInfo;
        }
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

void Graphics::Lighting::create(const std::filesystem::path& shadersPath, VkDevice device, VkRenderPass pRenderPass) {
    this->device = device;
    this->shadersPath = shadersPath;

    createDescriptorSetLayout();
    createPipeline(pRenderPass);
    createDescriptors();
}

void Graphics::Lighting::render(uint32_t frameNumber, VkCommandBuffer commandBuffer) const
{
    for(auto& lightSource: *lightSources){
        uint8_t mask = lightSource->getPipelineBitMask();
        lightSource->render(
            frameNumber,
            commandBuffer,
            {descriptorSets[frameNumber],
            (*depthMaps)[lightSource]->getDescriptorSets()[frameNumber]},
            pipelineLayoutMap.at(mask),
            pipelineMap.at(mask));
    }
}

}
