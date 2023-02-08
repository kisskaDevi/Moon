#ifndef DEVICE_H
#define DEVICE_H

#include <libs/vulkan/vulkan.h>

#include <vector>

struct device
{
    VkDevice                    instance{VK_NULL_HANDLE};
    VkPhysicalDeviceFeatures    deviceFeatures{};

    std::vector<VkQueue>        graphicsQueue;
    std::vector<VkQueue>        computeQueue;
    std::vector<VkQueue>        transferQueue;
    std::vector<VkQueue>        presentQueue;

    device(VkPhysicalDevice physicalDevice, uint32_t graphicsQueueCount, uint32_t computeQueueCount, uint32_t transferQueueCount, uint32_t presentQueueCount);
};

struct physicalDevice
{
    VkPhysicalDevice        instance{VK_NULL_HANDLE};

    std::vector<uint32_t>   graphicsQueueIndices;
    std::vector<uint32_t>   computeQueueIndices;
    std::vector<uint32_t>   transferQueueIndices;
    std::vector<uint32_t>   presentQueueIndices;

    std::vector<device>     logical;

    physicalDevice(VkPhysicalDevice physicalDevice);
    physicalDevice(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);

    void createDevice(device logical);

};

#endif // DEVICE_H
