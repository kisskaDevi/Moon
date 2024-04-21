#include "graphics.h"
#include "vkdefault.h"
#include "light.h"
#include "deferredAttachments.h"
#include "depthMap.h"
#include "operations.h"

void graphics::Lighting::Destroy(VkDevice device)
{
    for(auto& descriptorSetLayout: BufferDescriptorSetLayoutDictionary){
        if(descriptorSetLayout.second){ vkDestroyDescriptorSetLayout(device, descriptorSetLayout.second, nullptr); descriptorSetLayout.second = VK_NULL_HANDLE;}
    }
    for(auto& descriptorSetLayout: DescriptorSetLayoutDictionary){
        if(descriptorSetLayout.second){ vkDestroyDescriptorSetLayout(device, descriptorSetLayout.second, nullptr); descriptorSetLayout.second = VK_NULL_HANDLE;}
    }
    if(ShadowDescriptorSetLayout) {vkDestroyDescriptorSetLayout(device, ShadowDescriptorSetLayout, nullptr); ShadowDescriptorSetLayout = VK_NULL_HANDLE;}
    if(DescriptorSetLayout) {vkDestroyDescriptorSetLayout(device, DescriptorSetLayout, nullptr); DescriptorSetLayout = VK_NULL_HANDLE;}
    if(DescriptorPool)      {vkDestroyDescriptorPool(device, DescriptorPool, nullptr); DescriptorPool = VK_NULL_HANDLE;}

    for(auto& PipelineLayout: PipelineLayoutDictionary){
        if(PipelineLayout.second) {
            vkDestroyPipelineLayout(device, PipelineLayout.second, nullptr);
            PipelineLayout.second = VK_NULL_HANDLE;}
    }
    for(auto& Pipeline: PipelinesDictionary){
        if(Pipeline.second) {
            vkDestroyPipeline(device, Pipeline.second, nullptr);
            Pipeline.second = VK_NULL_HANDLE;}
    }
}

void graphics::Lighting::createDescriptorSetLayout(VkDevice device)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
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

    light::createBufferDescriptorSetLayout(device,&BufferDescriptorSetLayoutDictionary[lightType::spot]);
    light::createTextureDescriptorSetLayout(device,&DescriptorSetLayoutDictionary[lightType::spot]);
    depthMap::createDescriptorSetLayout(device, &ShadowDescriptorSetLayout);
}

void graphics::Lighting::createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass)
{
    std::filesystem::path spotVert = ShadersPath / "spotLightingPass/spotLightingVert.spv";
    std::filesystem::path spotFrag = ShadersPath / "spotLightingPass/spotLightingFrag.spv";
    createPipeline(lightType::spot, device, pInfo, pRenderPass, spotVert, spotFrag);
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
    CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &lighting.DescriptorPool));
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
    CHECK(vkAllocateDescriptorSets(device, &allocInfo, lighting.DescriptorSets.data()));
}

void graphics::updateLightingDescriptorSets(const buffersDatabase& bDatabase)
{
    for (uint32_t i = 0; i < image.Count; i++){
        std::vector<VkDescriptorImageInfo> imageInfos;
        imageInfos.push_back(VkDescriptorImageInfo{
            VK_NULL_HANDLE,
            deferredAttachments.GBuffer.position.instances[i].imageView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        });
        imageInfos.push_back(VkDescriptorImageInfo{
            VK_NULL_HANDLE,
            deferredAttachments.GBuffer.normal.instances[i].imageView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        });
        imageInfos.push_back(VkDescriptorImageInfo{
            VK_NULL_HANDLE,
            deferredAttachments.GBuffer.color.instances[i].imageView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        });
        imageInfos.push_back(VkDescriptorImageInfo{
            VK_NULL_HANDLE,
            deferredAttachments.GBuffer.depth.instances[i].imageView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        });

        VkDescriptorBufferInfo bufferInfo = bDatabase.descriptorBufferInfo(parameters.in.camera, i);

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

void graphics::Lighting::render(uint32_t frameNumber, VkCommandBuffer commandBuffer)
{
    for(auto& lightSource: *lightSources){
        uint8_t mask = lightSource->getPipelineBitMask();
        lightSource->render(frameNumber, commandBuffer, {DescriptorSets[frameNumber], (*depthMaps)[lightSource]->getDescriptorSets()[frameNumber]}, PipelineLayoutDictionary[mask], PipelinesDictionary[mask]);
    }
}
