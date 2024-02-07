#ifndef DEVICE_H
#define DEVICE_H

#include <vulkan.h>
#include <string>
#include <vector>
#include <map>

struct queueFamily{
    uint32_t index;
    VkQueueFlags flags;
    uint32_t queueCount;
    VkBool32 presentSupport;
    std::vector<float> queuePriorities;

    queueFamily() = default;
    queueFamily(uint32_t index, VkQueueFlags flag, uint32_t queueCount, VkBool32 presentSupport);
    queueFamily(const queueFamily& other);

    queueFamily& operator=(const queueFamily& other);

    bool availableQueueFlag(VkQueueFlags flag) const;
};

struct device
{
    VkDevice                    instance{VK_NULL_HANDLE};
    VkPhysicalDeviceFeatures    deviceFeatures{};

    std::map<uint32_t, std::vector<VkQueue>> queueMap;

    device() = default;
    device(VkPhysicalDeviceFeatures deviceFeatures):
        deviceFeatures(deviceFeatures)
    {}
};

struct physicalDevice
{
    VkPhysicalDevice instance{VK_NULL_HANDLE};
    VkPhysicalDeviceType type{VK_PHYSICAL_DEVICE_TYPE_MAX_ENUM};
    std::string name{};

    std::map<uint32_t, queueFamily> queueFamilies;
    std::vector<device> logical;
    std::vector<const char*> deviceExtensions;

#ifdef NDEBUG
    bool                    enableValidationLayers = false;
#else
    bool                    enableValidationLayers = true;
#endif
    const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

    physicalDevice() = default;
    physicalDevice(VkPhysicalDevice physicalDevice, std::vector<const char*> deviceExtensions = {});

    physicalDevice& operator=(const physicalDevice& other);
    physicalDevice(const physicalDevice& other);

    bool presentSupport(VkSurfaceKHR surface);
    VkResult createDevice(device logical, std::map<uint32_t,uint32_t> queueSizeMap);
    VkDevice& getLogical();
    const VkDevice& getLogical() const;
    VkQueue getQueue(uint32_t familyIndex, uint32_t queueIndex) const;
    bool createdLogical() const;
};

#endif // DEVICE_H
