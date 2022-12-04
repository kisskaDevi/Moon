#include "../graphics.h"
#include "core/operations.h"
#include "core/transformational/lightInterface.h"
#include "../../bufferObjects.h"

#include <array>
#include <iostream>

void deferredGraphics::Lighting::Destroy(VkDevice* device)
{
    for(auto& descriptorSetLayout: DescriptorSetLayoutDictionary){
        if(descriptorSetLayout.second) vkDestroyDescriptorSetLayout(*device, descriptorSetLayout.second, nullptr);
    }
    if(DescriptorSetLayout) vkDestroyDescriptorSetLayout(*device, DescriptorSetLayout, nullptr);
    if(DescriptorPool)      vkDestroyDescriptorPool(*device, DescriptorPool, nullptr);

    for(auto& PipelineLayout: PipelineLayoutDictionary){
        if(PipelineLayout.second) vkDestroyPipelineLayout(*device, PipelineLayout.second, nullptr);
    }
    for(auto& Pipeline: PipelinesDictionary){
        if(Pipeline.second) vkDestroyPipeline(*device, Pipeline.second, nullptr);
    }

    for (size_t i = 0; i < uniformBuffers.size(); i++)
    {
        if(uniformBuffers[i])           vkDestroyBuffer(*device, uniformBuffers[i], nullptr);
        if(uniformBuffersMemory[i])     vkFreeMemory(*device, uniformBuffersMemory[i], nullptr);
    }
}

void deferredGraphics::Lighting::createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount)
{
    uniformBuffers.resize(imageCount);
    uniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
        createBuffer(   physicalDevice,
                        device,
                        sizeof(UniformBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        uniformBuffers[i],
                        uniformBuffersMemory[i]);
}

void deferredGraphics::Lighting::createDescriptorSetLayout(VkDevice* device)
{
    uint32_t index = 0;

    std::array<VkDescriptorSetLayoutBinding,6> Binding{};
    for(index = 0; index<5;index++)
    {
        Binding.at(index).binding = index;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        Binding.at(index).descriptorCount = 1;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    }
        Binding.at(index).binding = index;
        Binding.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        Binding.at(index).descriptorCount = 1;
        Binding.at(index).stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        Binding.at(index).pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(Binding.size());
        layoutInfo.pBindings = Binding.data();
    vkCreateDescriptorSetLayout(*device, &layoutInfo, nullptr, &DescriptorSetLayout);

    createSpotLightDescriptorSetLayout(device,&DescriptorSetLayoutDictionary[0x0]);
}

void deferredGraphics::Lighting::createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass)
{
    std::string spotVert = ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\spotLightingPass\\spotLightingVert.spv";
    std::string spotFrag = ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\spotLightingPass\\spotLightingFrag.spv";
    std::string shadowSpotFrag = ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\spotLightingPass\\shadowSpotLightingFrag.spv";
    std::string scatteringSpotFrag = enableScattering ?
                ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\spotLightingPass\\scatteringSpotLightingFrag.spv":
                ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\spotLightingPass\\spotLightingFrag.spv";
    std::string scatteringShadowSpotFrag = enableScattering ?
                ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\spotLightingPass\\scatteringShadowSpotLightingFrag.spv":
                ExternalPath + "core\\graphics\\deferredGraphics\\shaders\\spotLightingPass\\shadowSpotLightingFrag.spv";
    createSpotPipeline(device,pInfo,pRenderPass,spotVert,spotFrag,&PipelineLayoutDictionary[(false<<5)|(false <<4)|(0x0)],&PipelinesDictionary[(false<<5)|(false <<4)|(0x0)]);
    createSpotPipeline(device,pInfo,pRenderPass,spotVert,shadowSpotFrag,&PipelineLayoutDictionary[(false<<5)|(true <<4)|(0x0)],&PipelinesDictionary[(false<<5)|(true <<4)|(0x0)]);
    createSpotPipeline(device,pInfo,pRenderPass,spotVert,scatteringSpotFrag,&PipelineLayoutDictionary[(true<<5)|(false <<4)|(0x0)],&PipelinesDictionary[(true<<5)|(false <<4)|(0x0)]);
    createSpotPipeline(device,pInfo,pRenderPass,spotVert,scatteringShadowSpotFrag,&PipelineLayoutDictionary[(true<<5)|(true <<4)|(0x0)],&PipelinesDictionary[(true<<5)|(true <<4)|(0x0)]);
}

void deferredGraphics::createLightingDescriptorPool()
{
    uint32_t index = 0;

    std::array<VkDescriptorPoolSize,6> poolSizes{};
        for(uint32_t i = 0;i<5;i++,index++)
            poolSizes[index] = {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, static_cast<uint32_t>(image.Count)};
        poolSizes[index] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(image.Count)};
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(image.Count);
    vkCreateDescriptorPool(*device, &poolInfo, nullptr, &lighting.DescriptorPool);
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
    vkAllocateDescriptorSets(*device, &allocInfo, lighting.DescriptorSets.data());
}

void deferredGraphics::updateLightingDescriptorSets()
{
    for (size_t i = 0; i < image.Count; i++)
    {
        uint32_t index = 0;

        std::array<VkDescriptorImageInfo,5> imageInfo{};
            for(index = 0; index<4;index++)
            {
                imageInfo.at(index).imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo.at(index).imageView = pAttachments.at(3+index)->imageView.at(i);
                imageInfo.at(index).sampler = VK_NULL_HANDLE;
            }
            imageInfo.at(index).imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.at(index).imageView = this->depthAttachment->imageView;
            imageInfo.at(index).sampler = VK_NULL_HANDLE;
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = lighting.uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

        std::array<VkWriteDescriptorSet,6> descriptorWrites{};
        for(index = 0; index<5;index++)
        {
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = lighting.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pImageInfo = &imageInfo.at(index);
        }
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = lighting.DescriptorSets.at(i);
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
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

void deferredGraphics::updateLightUbo(uint32_t imageIndex)
{
    for(auto lightSource: lighting.lightSources)
        lightSource->updateUniformBuffer(device,imageIndex);
}

void deferredGraphics::updateLightCmd(uint32_t imageIndex)
{
    std::vector<object*> objects(base.objects.size());

    uint32_t counter = 0;
    for(auto object: base.objects){
        objects[counter] = object;
        counter++;
    }

    for(auto lightSource: lighting.lightSources)
        if(lightSource->isShadowEnable())
            lightSource->updateShadowCommandBuffer(imageIndex,objects);
}

void deferredGraphics::getLightCommandbuffers(std::vector<VkCommandBuffer>& commandbufferSet, uint32_t imageIndex)
{
    for(auto lightSource: lighting.lightSources)
        if(lightSource->isShadowEnable())
            commandbufferSet.push_back(*lightSource->getShadowCommandBuffer(imageIndex));
}
