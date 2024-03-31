#include "rayTracingGraphics.h"

#include "swapChain.h"
#include "vkdefault.h"

#include <cstring>

void rayTracingGraphics::create()
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device.getLogical(), &poolInfo, nullptr, &commandPool);

    emptyTexture = createEmptyTexture(device, commandPool);

    finalAttachment.create(
        device.instance,
        device.getLogical(),
        format,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        extent,
        imageCount
    );
    VkSamplerCreateInfo SamplerInfo = vkDefault::samler();
    vkCreateSampler(device.getLogical(), &SamplerInfo, nullptr, &finalAttachment.sampler);

    imageInfo bbInfo{
        imageCount,
        format,
        extent,
        VK_SAMPLE_COUNT_1_BIT
    };

    bbGraphics.create(device.instance, device.getLogical(), bbInfo, shadersPath);

    imageInfo swapChainInfo{
        imageCount,
        format,
        swapChainKHR->getExtent(),
        VK_SAMPLE_COUNT_1_BIT
    };

    Link.setEmptyTexture(emptyTexture);
    Link.setImageCount(imageCount);
    Link.setDeviceProp(device.getLogical());
    Link.setShadersPath(shadersPath);
    Link.createDescriptorSetLayout();
    Link.createPipeline(&swapChainInfo);
    Link.createDescriptorPool();
    Link.createDescriptorSets();
    Link.updateDescriptorSets(&finalAttachment, &bbGraphics.getAttachments());

    Buffer::create(
        device.instance,
        device.getLogical(),
        sizeof(uint32_t) * extent.width * extent.height,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &stagingBuffer.instance,
        &stagingBuffer.memory
    );

    hostFrameBuffer = new uint32_t[extent.width * extent.height];
    vkMapMemory(device.getLogical(), stagingBuffer.memory, 0, sizeof(uint32_t) * extent.width * extent.height, 0, &stagingBuffer.map);

    rayTracer.create();
}

void rayTracingGraphics::destroy() {
    if(emptyTexture){
        emptyTexture->destroy(device.getLogical());
        delete emptyTexture;
    }
    if(hostFrameBuffer){
        delete[] hostFrameBuffer;
        hostFrameBuffer = nullptr;
    }

    stagingBuffer.destroy(device.getLogical());
    if(commandPool) {vkDestroyCommandPool(device.getLogical(), commandPool, nullptr); commandPool = VK_NULL_HANDLE;}
    finalAttachment.deleteAttachment(device.getLogical());
    finalAttachment.deleteSampler(device.getLogical());
    Link.destroy();
    bbGraphics.destroy();
}

std::vector<std::vector<VkSemaphore>> rayTracingGraphics::submit(const std::vector<std::vector<VkSemaphore>>&, const std::vector<VkFence>&, uint32_t imageIndex)
{
    rayTracer.calculateImage(hostFrameBuffer);

    std::memcpy(stagingBuffer.map, hostFrameBuffer, sizeof(uint32_t) * extent.width * extent.height);

    VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
    Texture::transitionLayout(commandBuffer, finalAttachment.instances[imageIndex].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    Texture::copy(commandBuffer, stagingBuffer.instance, finalAttachment.instances[imageIndex].image, {extent.width, extent.height, 1}, 1);
    Texture::transitionLayout(commandBuffer, finalAttachment.instances[imageIndex].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
    bbGraphics.render(commandBuffer, imageIndex);
    SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool, &commandBuffer);

    return std::vector<std::vector<VkSemaphore>>();
}

void rayTracingGraphics::update(uint32_t imageIndex) {
    rayTracer.update();
    bbGraphics.update(imageIndex);
}
