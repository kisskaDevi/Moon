#include "rayTracingGraphics.h"

#include "swapChain.h"
#include "vkdefault.h"

#include <cstring>

namespace moon::rayTracingGraphics {

void RayTracingGraphics::ImageResource::create(const std::string& id, moon::utils::PhysicalDevice phDevice, VkFormat format, VkExtent2D extent, uint32_t imageCount){
    this->id = id;

    host = new uint32_t[extent.width * extent.height];

    moon::utils::buffer::create(
        phDevice.instance,
        phDevice.getLogical(),
        sizeof(uint32_t) * extent.width * extent.height,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &hostDevice.instance,
        &hostDevice.memory
        );
    vkMapMemory(phDevice.getLogical(), hostDevice.memory, 0, sizeof(uint32_t) * extent.width * extent.height, 0, &hostDevice.map);

    device.create(
        phDevice.instance,
        phDevice.getLogical(),
        format,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        extent,
        imageCount
        );
    VkSamplerCreateInfo SamplerInfo = moon::utils::vkDefault::samler();
    vkCreateSampler(phDevice.getLogical(), &SamplerInfo, nullptr, &device.sampler);
}

void RayTracingGraphics::ImageResource::destroy(moon::utils::PhysicalDevice phDevice){
    if(host){
        delete[] host;
        host = nullptr;
    }
    hostDevice.destroy(phDevice.getLogical());
    device.deleteAttachment(phDevice.getLogical());
    device.deleteSampler(phDevice.getLogical());
}

void RayTracingGraphics::ImageResource::moveFromHostToHostDevice(VkExtent2D extent){
    std::memcpy(hostDevice.map, host, sizeof(uint32_t) * extent.width * extent.height);
}

void RayTracingGraphics::ImageResource::copyToDevice(VkCommandBuffer commandBuffer, VkExtent2D extent, uint32_t imageIndex){
    moon::utils::texture::transitionLayout(commandBuffer, device.instances[imageIndex].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    moon::utils::texture::copy(commandBuffer, hostDevice.instance, device.instances[imageIndex].image, {extent.width, extent.height, 1}, 1);
    moon::utils::texture::transitionLayout(commandBuffer, device.instances[imageIndex].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
}

void RayTracingGraphics::create()
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device.getLogical(), &poolInfo, nullptr, &commandPool);

    emptyTexture = createEmptyTexture(device, commandPool);
    aDatabase.addEmptyTexture("black", emptyTexture);

    color.create("color", device, format, extent, imageCount);
    aDatabase.addAttachmentData(color.id, true, &color.device);

    bloom.create("bloom", device, format, extent, imageCount);
    aDatabase.addAttachmentData(bloom.id, true, &bloom.device);

    moon::utils::ImageInfo bloomInfo{imageCount, format, extent, VK_SAMPLE_COUNT_1_BIT};

    moon::workflows::BloomParameters bloomParams;
    bloomParams.in.bloom = bloom.id;
    bloomParams.out.bloom = "finalBloom";

    bloomGraph = moon::workflows::BloomGraphics(bloomParams, bloomEnable, 8, VK_IMAGE_LAYOUT_UNDEFINED);
    bloomGraph.setShadersPath(workflowsShadersPath);
    bloomGraph.setDeviceProp(device.instance, device.getLogical());
    bloomGraph.setImageProp(&bloomInfo);
    bloomGraph.create(aDatabase);
    bloomGraph.createCommandBuffers(commandPool);
    bloomGraph.updateDescriptorSets(bDatabase, aDatabase);

    moon::utils::ImageInfo bbInfo{imageCount, format, extent, VK_SAMPLE_COUNT_1_BIT};
    std::string bbId = "bb";
    bbGraphics.create(device.instance, device.getLogical(), bbInfo, shadersPath);
    aDatabase.addAttachmentData(bbId, bbGraphics.getEnable(), &bbGraphics.getAttachments());

    moon::utils::ImageInfo swapChainInfo{ imageCount, format, swapChainKHR->getExtent(), VK_SAMPLE_COUNT_1_BIT};
    RayTracingLinkParameters linkParams;
    linkParams.in.color = color.id;
    linkParams.in.bloom = bloomParams.out.bloom;
    linkParams.in.boundingBox = bbId;

    Link.setParameters(linkParams);
    Link.setImageCount(imageCount);
    Link.setDeviceProp(device.getLogical());
    Link.setShadersPath(shadersPath);
    Link.createDescriptorSetLayout();
    Link.createPipeline(&swapChainInfo);
    Link.createDescriptorPool();
    Link.createDescriptorSets();
    Link.updateDescriptorSets(aDatabase);

    rayTracer.create();
}

void RayTracingGraphics::destroy() {
    if(emptyTexture){
        emptyTexture->destroy(device.getLogical());
        delete emptyTexture;
    }

    color.destroy(device);
    bloom.destroy(device);

    bloomGraph.destroy();

    if(commandPool) {vkDestroyCommandPool(device.getLogical(), commandPool, nullptr); commandPool = VK_NULL_HANDLE;}
    Link.destroy();
    bbGraphics.destroy();
    aDatabase.destroy();
}

std::vector<std::vector<VkSemaphore>> RayTracingGraphics::submit(const std::vector<std::vector<VkSemaphore>>&, const std::vector<VkFence>&, uint32_t imageIndex)
{
    rayTracer.calculateImage(color.host, bloom.host);

    color.moveFromHostToHostDevice(extent);
    bloom.moveFromHostToHostDevice(extent);

    std::vector<VkCommandBuffer> commandBuffers;
    commandBuffers.push_back(moon::utils::singleCommandBuffer::create(device.getLogical(),commandPool));
    color.copyToDevice(commandBuffers.back(), extent, imageIndex);
    bloom.copyToDevice(commandBuffers.back(), extent, imageIndex);
    bbGraphics.render(commandBuffers.back(), imageIndex);
    moon::utils::singleCommandBuffer::submit(device.getLogical(), device.getQueue(0,0), commandPool, commandBuffers.size(), commandBuffers.data());

    bloomGraph.beginCommandBuffer(imageIndex);
    bloomGraph.updateCommandBuffer(imageIndex);
    bloomGraph.endCommandBuffer(imageIndex);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &bloomGraph.getCommandBuffer(imageIndex);
    VkResult result = vkQueueSubmit(device.getQueue(0,0), 1, &submitInfo, VK_NULL_HANDLE);
    CHECK(result);

    result = vkQueueWaitIdle(device.getQueue(0,0));
    CHECK(result);

    return std::vector<std::vector<VkSemaphore>>();
}

void RayTracingGraphics::update(uint32_t imageIndex) {
    rayTracer.update();
    bbGraphics.update(imageIndex);
}

}
