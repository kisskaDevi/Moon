#include "../graphics.h"
#include "../../../transformational/camera.h"
#include "../../../transformational/lightInterface.h"

#include <array>
#include <iostream>

struct lightPassPushConst{
    alignas(4) float                minAmbientFactor;
};

void deferredGraphics::Lighting::Destroy(VkDevice* device)
{
    for(auto& descriptorSetLayout: DescriptorSetLayoutDictionary){
        if(descriptorSetLayout.second){ vkDestroyDescriptorSetLayout(*device, descriptorSetLayout.second, nullptr); descriptorSetLayout.second = VK_NULL_HANDLE;}
    }
    if(DescriptorSetLayout) {vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr); DescriptorSetLayout = VK_NULL_HANDLE;}
    if(DescriptorPool)      {vkDestroyDescriptorPool(*device, DescriptorPool, nullptr); DescriptorPool = VK_NULL_HANDLE;}

    for(auto& PipelineLayout: PipelineLayoutDictionary){
        if(PipelineLayout.second) {vkDestroyPipelineLayout(*device, PipelineLayout.second, nullptr); PipelineLayout.second = VK_NULL_HANDLE;}
    }
    for(auto& Pipeline: PipelinesDictionary){
        if(Pipeline.second) {vkDestroyPipeline(*device, Pipeline.second, nullptr);  Pipeline.second = VK_NULL_HANDLE;}
    }
}

void deferredGraphics::Lighting::createDescriptorSetLayout(VkDevice* device)
{
    std::vector<VkDescriptorSetLayoutBinding> binding;
    for(uint32_t index = 0; index<5;index++)
    {
        binding.push_back(VkDescriptorSetLayoutBinding{});
            binding.back().binding = binding.size() - 1;
            binding.back().descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            binding.back().descriptorCount = 1;
            binding.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            binding.back().pImmutableSamplers = nullptr;
    }
        binding.push_back(VkDescriptorSetLayoutBinding{});
            binding.back().binding = binding.size() - 1;
            binding.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.back().descriptorCount = 1;
            binding.back().stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            binding.back().pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        layoutInfo.pBindings = binding.data();
    vkCreateDescriptorSetLayout(*device, &layoutInfo, nullptr, &DescriptorSetLayout);

    SpotLight::createDescriptorSetLayout(*device,&DescriptorSetLayoutDictionary[0x0]);
}

void deferredGraphics::Lighting::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    std::string spotVert = ExternalPath + "core\\deferredGraphics\\shaders\\spotLightingPass\\spotLightingVert.spv";
    std::string spotFrag = ExternalPath + "core\\deferredGraphics\\shaders\\spotLightingPass\\spotLightingFrag.spv";
    std::string shadowSpotFrag = ExternalPath + "core\\deferredGraphics\\shaders\\spotLightingPass\\shadowSpotLightingFrag.spv";
    std::string scatteringSpotFrag = enableScattering ?
                ExternalPath + "core\\deferredGraphics\\shaders\\spotLightingPass\\scatteringSpotLightingFrag.spv":
                ExternalPath + "core\\deferredGraphics\\shaders\\spotLightingPass\\spotLightingFrag.spv";
    std::string scatteringShadowSpotFrag = enableScattering ?
                ExternalPath + "core\\deferredGraphics\\shaders\\spotLightingPass\\scatteringShadowSpotLightingFrag.spv":
                ExternalPath + "core\\deferredGraphics\\shaders\\spotLightingPass\\shadowSpotLightingFrag.spv";
    createSpotPipeline(device,pInfo,pRenderPass,spotVert,spotFrag,&PipelineLayoutDictionary[(false<<5)|(false <<4)|(0x0)],&PipelinesDictionary[(false<<5)|(false <<4)|(0x0)]);
    createSpotPipeline(device,pInfo,pRenderPass,spotVert,shadowSpotFrag,&PipelineLayoutDictionary[(false<<5)|(true <<4)|(0x0)],&PipelinesDictionary[(false<<5)|(true <<4)|(0x0)]);
    createSpotPipeline(device,pInfo,pRenderPass,spotVert,scatteringSpotFrag,&PipelineLayoutDictionary[(true<<5)|(false <<4)|(0x0)],&PipelinesDictionary[(true<<5)|(false <<4)|(0x0)]);
    createSpotPipeline(device,pInfo,pRenderPass,spotVert,scatteringShadowSpotFrag,&PipelineLayoutDictionary[(true<<5)|(true <<4)|(0x0)],&PipelinesDictionary[(true<<5)|(true <<4)|(0x0)]);
}

void deferredGraphics::createLightingDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    for(uint32_t i = 0; i < 5 ;i++){
        poolSizes.push_back(VkDescriptorPoolSize{});
            poolSizes.back() = {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, static_cast<uint32_t>(image.Count)};
    }
        poolSizes.push_back(VkDescriptorPoolSize{});
            poolSizes.back() = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(image.Count)};

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &lighting.DescriptorPool);
}

void deferredGraphics::createLightingDescriptorSets()
{
    lighting.DescriptorSets.resize(image.Count);
    std::vector<VkDescriptorSetLayout> layouts(image.Count, lighting.DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = lighting.DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(image.Count);
        allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(device, &allocInfo, lighting.DescriptorSets.data());
}

void deferredGraphics::updateLightingDescriptorSets(camera* cameraObject)
{
    for (size_t i = 0; i < image.Count; i++)
    {
        std::vector<VkDescriptorImageInfo> imageInfo;
        for(uint32_t index = 0; index < 4;index++)
        {
            imageInfo.push_back(VkDescriptorImageInfo{});
            imageInfo.back().imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.back().imageView = pAttachments[DeferredAttachments::getGBufferOffset() + index]->imageView[i];
            imageInfo.back().sampler = VK_NULL_HANDLE;
        }
            imageInfo.push_back(VkDescriptorImageInfo{});
            imageInfo.back().imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.back().imageView = this->pAttachments[DeferredAttachments::getGBufferOffset() - 1]->imageView[i];
            imageInfo.back().sampler = VK_NULL_HANDLE;

        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = cameraObject->getBuffer(i);
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        for(uint32_t index = 0; index<5;index++)
        {
            descriptorWrites.push_back(VkWriteDescriptorSet{});
                descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites.back().dstSet = lighting.DescriptorSets[i];
                descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
                descriptorWrites.back().dstArrayElement = 0;
                descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
                descriptorWrites.back().descriptorCount = 1;
                descriptorWrites.back().pImageInfo = &imageInfo[index];
        }
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = lighting.DescriptorSets[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void deferredGraphics::Lighting::render(uint32_t frameNumber, VkCommandBuffer commandBuffers)
{
    for(auto& lightSource: lightSources)
    {
        vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelinesDictionary[lightSource->getPipelineBitMask()]);

        std::vector<VkDescriptorSet> descriptorSets = {DescriptorSets[frameNumber],lightSource->getDescriptorSets()[frameNumber]};
        vkCmdBindDescriptorSets(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayoutDictionary[lightSource->getPipelineBitMask()], 0, static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);

        vkCmdDraw(commandBuffers, 18, 1, 0, 0);
    }
}

void deferredGraphics::updateLightSourcesUniformBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    for(auto lightSource: lighting.lightSources)
        lightSource->updateUniformBuffer(commandBuffer,imageIndex);
}
