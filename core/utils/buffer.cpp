#include "buffer.h"
#include "operations.h"
#include <cstring>

namespace moon::utils {

void Buffer::destroy(VkDevice device){
    if(map){
        vkUnmapMemory(device, memory);
        map = nullptr;
    }
    buffer::destroy(device, instance, memory);
}

void destroyBuffers(VkDevice device, std::vector<Buffer>& uniformBuffers){
    for(auto& buffer: uniformBuffers){
        buffer.destroy(device);
    }
    uniformBuffers.clear();
};

void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, size_t bufferSize, void* data, VkBufferUsageFlagBits usage, Buffer& staging, Buffer& deviceLocal)
{
    buffer::create(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &staging.instance, &staging.memory);
    buffer::create(physicalDevice, device, bufferSize, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &deviceLocal.instance, &deviceLocal.memory);

    Memory::instance().nameMemory(staging.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", createBuffer, staging");
    Memory::instance().nameMemory(deviceLocal.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", createBuffer, deviceLocal");

    CHECK(vkMapMemory(device, staging.memory, 0, bufferSize, 0, &staging.map));
        std::memcpy(staging.map, data, bufferSize);
    vkUnmapMemory(device, staging.memory);
    staging.map = nullptr;

    buffer::copy(commandBuffer, bufferSize, staging.instance, deviceLocal.instance);
};

void Buffers::create(VkPhysicalDevice                physicalDevice,
                     VkDevice                        device,
                     VkDeviceSize                    size,
                     VkBufferUsageFlags              usage,
                     VkMemoryPropertyFlags           properties,
                     size_t                          instancesCount)
{
    instances.resize(instancesCount);
    for (auto& buffer: instances){
        buffer::create(physicalDevice,
                       device,
                       size,
                       usage,
                       properties,
                       &buffer.instance,
                       &buffer.memory);
        buffer.size = size;
    }
}

void Buffers::map(VkDevice device)
{
    for (auto& buffer: instances){
        CHECK(vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &buffer.map));
    }
}

void Buffers::copy(size_t imageIndex, void *data)
{
    std::memcpy(instances[imageIndex].map, data, instances[imageIndex].size);
}

void Buffers::destroy(VkDevice device)
{
    for (auto& buffer: instances){
        buffer.destroy(device);
        if(buffer.map){
            vkUnmapMemory(device, buffer.memory);
        }
    }
    instances.clear();
}

void BuffersDatabase::destroy()
{
    buffersMap.clear();
}

bool BuffersDatabase::addBufferData(const std::string& id, const Buffers* pBuffer)
{
    if(buffersMap.count(id) > 0) return false;

    buffersMap[id] = pBuffer;
    return true;
}

const Buffers* BuffersDatabase::get(const std::string& id) const
{
    return buffersMap.count(id) > 0 ? buffersMap.at(id) : nullptr;
}

VkBuffer BuffersDatabase::buffer(const std::string& id, const uint32_t imageIndex) const
{
    return buffersMap.count(id) > 0 && buffersMap.at(id) ? buffersMap.at(id)->instances[imageIndex].instance : VK_NULL_HANDLE;
}

VkDescriptorBufferInfo BuffersDatabase::descriptorBufferInfo(const std::string& id, const uint32_t imageIndex) const
{
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffersMap.at(id)->instances[imageIndex].instance;
    bufferInfo.offset = 0;
    bufferInfo.range = buffersMap.at(id)->instances[imageIndex].size;
    return bufferInfo;
}

}
