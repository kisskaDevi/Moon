#ifndef BUFFER_H
#define BUFFER_H

#include <vulkan.h>
#include <vector>
#include <string>
#include <unordered_map>

namespace moon::utils {

struct Buffer{
    VkBuffer            instance{VK_NULL_HANDLE};
    VkDeviceMemory      memory{VK_NULL_HANDLE};
    bool                updateFlag{true};
    void*               map{nullptr};
    size_t              size{0};

    Buffer() = default;
    ~Buffer() = default;
    void destroy(VkDevice device);
};

void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, size_t bufferSize, void* data, VkBufferUsageFlagBits usage, Buffer& staging, Buffer& deviceLocal);
void destroyBuffers(VkDevice device, std::vector<Buffer>& uniformBuffers);


struct Buffers{
    std::vector<Buffer> instances;

    void create(VkPhysicalDevice                physicalDevice,
                VkDevice                        device,
                VkDeviceSize                    size,
                VkBufferUsageFlags              usage,
                VkMemoryPropertyFlags           properties,
                size_t                          instancesCount);
    void map(VkDevice device);
    void copy(size_t imageIndex, void* data);
    void destroy(VkDevice device);
};

struct BuffersDatabase{
    std::unordered_map<std::string, const Buffers*> buffersMap;

    BuffersDatabase() = default;
    BuffersDatabase(const BuffersDatabase&) = default;
    BuffersDatabase& operator=(const BuffersDatabase&) = default;

    void destroy();

    bool addBufferData(const std::string& id, const Buffers* pBuffer);
    const Buffers* get(const std::string& id) const;
    VkBuffer buffer(const std::string& id, const uint32_t imageIndex) const;
    VkDescriptorBufferInfo descriptorBufferInfo(const std::string& id, const uint32_t imageIndex) const;
};

}
#endif // BUFFER_H
