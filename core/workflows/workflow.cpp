#include "workflow.h"
#include "operations.h"

namespace moon::workflows {

Workflow& Workflow::setDeviceProp(VkPhysicalDevice physicalDevice, VkDevice device){
    this->physicalDevice = physicalDevice;
    this->device = device;
    return *this;
}

void Workflow::createCommandBuffers(VkCommandPool commandPool)
{
    commandBuffers.resize(imageInfo.Count);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(imageInfo.Count);
    CHECK(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()));
}

void Workflow::beginCommandBuffer(uint32_t frameNumber){
    CHECK(vkResetCommandBuffer(commandBuffers[frameNumber],0));

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

    CHECK(vkBeginCommandBuffer(commandBuffers[frameNumber], &beginInfo));
}

void Workflow::endCommandBuffer(uint32_t frameNumber){
    CHECK(vkEndCommandBuffer(commandBuffers[frameNumber]));
}

VkCommandBuffer& Workflow::getCommandBuffer(uint32_t frameNumber)
{
    return commandBuffers[frameNumber];
}

void Workflow::freeCommandBuffer(VkCommandPool commandPool){
    if(commandBuffers.data()){
        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
    }
    commandBuffers.clear();
}

}
