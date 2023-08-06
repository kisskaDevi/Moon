#include "graphics.h"
#include "vkdefault.h"
#include "camera.h"
#include "light.h"

#include <iostream>

struct lightPassPushConst{
    alignas(4) float minAmbientFactor;
};

void graphics::Lighting::Destroy(VkDevice device)
{
    for(auto& descriptorSetLayout: BufferDescriptorSetLayoutDictionary){
        if(descriptorSetLayout.second){ vkDestroyDescriptorSetLayout(device, descriptorSetLayout.second, nullptr); descriptorSetLayout.second = VK_NULL_HANDLE;}
    }
    for(auto& descriptorSetLayout: DescriptorSetLayoutDictionary){
        if(descriptorSetLayout.second){ vkDestroyDescriptorSetLayout(device, descriptorSetLayout.second, nullptr); descriptorSetLayout.second = VK_NULL_HANDLE;}
    }
    if(DescriptorSetLayout) {vkDestroyDescriptorSetLayout(device, DescriptorSetLayout, nullptr); DescriptorSetLayout = VK_NULL_HANDLE;}
    if(DescriptorPool)      {vkDestroyDescriptorPool(device, DescriptorPool, nullptr); DescriptorPool = VK_NULL_HANDLE;}

    for(auto& PipelineLayout: PipelineLayoutDictionary){
        if(PipelineLayout.second) {vkDestroyPipelineLayout(device, PipelineLayout.second, nullptr); PipelineLayout.second = VK_NULL_HANDLE;}
    }
    for(auto& Pipeline: PipelinesDictionary){
        if(Pipeline.second) {vkDestroyPipeline(device, Pipeline.second, nullptr);  Pipeline.second = VK_NULL_HANDLE;}
    }
}

void graphics::Lighting::createDescriptorSetLayout(VkDevice device)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &DescriptorSetLayout);

    light::createBufferDescriptorSetLayout(device,&BufferDescriptorSetLayoutDictionary[0x0]);
    light::createTextureDescriptorSetLayout(device,&DescriptorSetLayoutDictionary[0x0]);
}

void graphics::Lighting::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass)
{
    std::filesystem::path spotVert = ShadersPath / "spotLightingPass/spotLightingVert.spv";
    std::filesystem::path spotFrag = ShadersPath / "spotLightingPass/spotLightingFrag.spv";
    createSpotPipeline(device,pInfo,pRenderPass,spotVert,spotFrag,false,false);
    createSpotPipeline(device,pInfo,pRenderPass,spotVert,spotFrag,true,false);
    createSpotPipeline(device,pInfo,pRenderPass,spotVert,spotFrag,false,true);
    createSpotPipeline(device,pInfo,pRenderPass,spotVert,spotFrag,true,true);
}

void graphics::createLightingDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, static_cast<uint32_t>(5 * image.Count)});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(image.Count)});

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &lighting.DescriptorPool);
}

void graphics::createLightingDescriptorSets()
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

void graphics::updateLightingDescriptorSets(camera* cameraObject)
{
    for (uint32_t i = 0; i < image.Count; i++)
    {
        std::vector<VkDescriptorImageInfo> imageInfos;
        for(uint32_t index = 0; index < 4;index++)
        {
            imageInfos.push_back(VkDescriptorImageInfo{});
            imageInfos.back().imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfos.back().imageView = pAttachments[DeferredAttachments::getGBufferOffset() + index]->imageView[i];
            imageInfos.back().sampler = VK_NULL_HANDLE;
        }
            imageInfos.push_back(VkDescriptorImageInfo{});
            imageInfos.back().imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfos.back().imageView = this->pAttachments[DeferredAttachments::getGBufferOffset() - 1]->imageView[i];
            imageInfos.back().sampler = VK_NULL_HANDLE;

        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = cameraObject->getBuffer(i);
            bufferInfo.offset = 0;
            bufferInfo.range = cameraObject->getBufferRange();

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        for(auto& imageInfo: imageInfos){
            descriptorWrites.push_back(VkWriteDescriptorSet{});
                descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites.back().dstSet = lighting.DescriptorSets[i];
                descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
                descriptorWrites.back().dstArrayElement = 0;
                descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
                descriptorWrites.back().descriptorCount = 1;
                descriptorWrites.back().pImageInfo = &imageInfo;
        }
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = lighting.DescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void graphics::Lighting::render(uint32_t frameNumber, VkCommandBuffer commandBuffers)
{
    for(auto& lightSource: lightSources)
    {
        vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelinesDictionary[lightSource->getPipelineBitMask()]);

        std::vector<VkDescriptorSet> descriptorSets = {
            DescriptorSets[frameNumber],
            lightSource->getBufferDescriptorSets()[frameNumber],
            lightSource->getDescriptorSets()[frameNumber]};
        vkCmdBindDescriptorSets(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayoutDictionary[lightSource->getPipelineBitMask()], 0, static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);

        vkCmdDraw(commandBuffers, 18, 1, 0, 0);
    }
}

void graphics::updateLightSourcesUniformBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    for(auto lightSource: lighting.lightSources)
        lightSource->updateUniformBuffer(commandBuffer,imageIndex);
}

