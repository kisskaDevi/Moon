#ifndef BUFFER_H
#define BUFFER_H

#include <vulkan.h>

class buffer
{
public:
    VkBuffer       instance{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    bool           updateFlag{true};
    void*          map{nullptr};

    buffer() = default;
    ~buffer() = default;
    void destroy(VkDevice device);
};

#endif // BUFFER_H
