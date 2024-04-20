#ifndef BUFFER_H
#define BUFFER_H

#include <vulkan.h>
#include <vector>
#include <string>
#include <unordered_map>

struct buffer{
    VkBuffer            instance{VK_NULL_HANDLE};
    VkDeviceMemory      memory{VK_NULL_HANDLE};
    bool                updateFlag{true};
    void*               map{nullptr};
    size_t              size{0};

    buffer() = default;
    ~buffer() = default;
    void destroy(VkDevice device);
};

void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, size_t bufferSize, void* data, VkBufferUsageFlagBits usage, buffer& staging, buffer& deviceLocal);
void destroyBuffers(VkDevice device, std::vector<buffer>& uniformBuffers);


struct buffers{
    std::vector<buffer> instances;

    void create(VkPhysicalDevice                physicalDevice,
                VkDevice                        device,
                VkDeviceSize                    size,
                VkBufferUsageFlags              usage,
                VkMemoryPropertyFlags           properties,
                size_t                          instancesCount);
    void map(VkDevice device);
    void destroy(VkDevice device);
};

struct buffersDatabase{
    std::unordered_map<std::string, const buffers*> buffersMap;

    buffersDatabase() = default;
    buffersDatabase(const buffersDatabase&) = default;
    buffersDatabase& operator=(const buffersDatabase&) = default;

    void destroy();

    bool addBufferData(const std::string& id, const buffers* pBuffer);
    const buffers* get(const std::string& id) const;
    VkBuffer buffer(const std::string& id, const uint32_t imageIndex) const;
    VkDescriptorBufferInfo descriptorBufferInfo(const std::string& id, const uint32_t imageIndex) const;
};

#endif // BUFFER_H
