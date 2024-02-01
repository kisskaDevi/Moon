#ifndef BUFFER_H
#define BUFFER_H

#include <vulkan.h>
#include <vector>

struct buffer
{
    VkBuffer            instance{VK_NULL_HANDLE};
    VkDeviceMemory      memory{VK_NULL_HANDLE};
    bool                updateFlag{true};
    void*               map{nullptr};

    buffer() = default;
    ~buffer() = default;
    void destroy(VkDevice device);
};

void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, size_t bufferSize, void* data, VkBufferUsageFlagBits usage, buffer& staging, buffer& deviceLocal);
void destroyBuffers(VkDevice device, std::vector<buffer>& uniformBuffers);

#endif // BUFFER_H
