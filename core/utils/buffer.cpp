#include "buffer.h"
#include "operations.h"
#include <cstring>

void buffer::destroy(VkDevice device){
    if(map){
        vkUnmapMemory(device, memory);
        map = nullptr;
    }
    Buffer::destroy(device, instance, memory);
}

void destroyBuffers(VkDevice device, std::vector<buffer>& uniformBuffers){
    for(auto& buffer: uniformBuffers){
        buffer.destroy(device);
    }
    uniformBuffers.clear();
};

void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, size_t bufferSize, void* data, VkBufferUsageFlagBits usage, buffer& staging, buffer& deviceLocal)
{
    Buffer::create(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &staging.instance, &staging.memory);
    Buffer::create(physicalDevice, device, bufferSize, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &deviceLocal.instance, &deviceLocal.memory);

    Memory::instance().nameMemory(staging.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", createBuffer, staging");
    Memory::instance().nameMemory(deviceLocal.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", createBuffer, deviceLocal");

    CHECK(vkMapMemory(device, staging.memory, 0, bufferSize, 0, &staging.map));
        std::memcpy(staging.map, data, bufferSize);
    vkUnmapMemory(device, staging.memory);
    staging.map = nullptr;

    Buffer::copy(commandBuffer, bufferSize, staging.instance, deviceLocal.instance);
};
