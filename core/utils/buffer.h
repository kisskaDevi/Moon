#ifndef BUFFER_H
#define BUFFER_H

#include <vulkan.h>
#include <vector>
#include <string>
#include <unordered_map>

#include "vkdefault.h"

namespace moon::utils {

using Buffer = vkDefault::Buffer;
using Buffers = std::vector<Buffer>;

void createDeviceBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, size_t bufferSize, void* data, VkBufferUsageFlagBits usage, Buffer& cache, Buffer& deviceLocal);

struct BuffersDatabase{
    std::unordered_map<std::string, const Buffers*> buffersMap;

    BuffersDatabase() = default;
    BuffersDatabase(const BuffersDatabase&) = default;
    BuffersDatabase& operator=(const BuffersDatabase&) = default;

    bool add(const std::string& id, const Buffers* pBuffer);
    bool remove(const std::string& id);
    const Buffers* get(const std::string& id) const;
    VkBuffer buffer(const std::string& id, const uint32_t imageIndex) const;
    VkDescriptorBufferInfo descriptorBufferInfo(const std::string& id, const uint32_t imageIndex) const;
};

}
#endif // BUFFER_H
