#ifndef DEVICE_H
#define DEVICE_H

#include <vulkan.h>
#include <string>
#include <vector>
#include <map>

namespace moon::utils {

struct QueueFamily{
    uint32_t index;
    VkQueueFlags flags;
    uint32_t queueCount;
    VkBool32 presentSupport;
    std::vector<float> queuePriorities;

    QueueFamily() = default;
    QueueFamily(uint32_t index, VkQueueFlags flag, uint32_t queueCount, VkBool32 presentSupport);
    QueueFamily(const QueueFamily& other);

    QueueFamily& operator=(const QueueFamily& other);

    bool availableQueueFlag(VkQueueFlags flag) const;
};

struct Device{
    VkDevice instance{VK_NULL_HANDLE};
    VkPhysicalDeviceFeatures deviceFeatures{};
    std::map<uint32_t, std::vector<VkQueue>> queueMap;

    Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&& other);
    Device& operator=(Device&& other);
    void swap(Device& other);
    Device(VkPhysicalDeviceFeatures deviceFeatures);
    ~Device();
};

using DeviceIndex = uint32_t;

struct PhysicalDeviceProperties{
    DeviceIndex index{0x7FFFFFFF};
    VkPhysicalDeviceType type{VK_PHYSICAL_DEVICE_TYPE_MAX_ENUM};
    std::string name{};
};

struct PhysicalDevice{
    VkPhysicalDevice instance{VK_NULL_HANDLE};
    PhysicalDeviceProperties properties{};

    std::map<uint32_t, QueueFamily> queueFamilies;
    std::vector<Device> logical;
    std::vector<const char*> deviceExtensions;

#ifdef NDEBUG
    bool enableValidationLayers = false;
#else
    bool enableValidationLayers = true;
#endif
    const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

    PhysicalDevice() = default;
    PhysicalDevice(VkPhysicalDevice physicalDevice, std::vector<const char*> deviceExtensions = {});

    PhysicalDevice& operator=(const PhysicalDevice& other) = delete;
    PhysicalDevice(const PhysicalDevice& other) = delete;
    PhysicalDevice& operator=(PhysicalDevice&& other);
    PhysicalDevice(PhysicalDevice&& other);
    void swap(PhysicalDevice& other);

    bool presentSupport(VkSurfaceKHR surface);
    VkResult createDevice(VkPhysicalDeviceFeatures deviceFeatures, std::map<uint32_t,uint32_t> queueSizeMap);
    VkDevice& getLogical();
    const VkDevice& getLogical() const;
    VkQueue getQueue(uint32_t familyIndex, uint32_t queueIndex) const;
    bool createdLogical() const;
};

using PhysicalDeviceMap = std::map<DeviceIndex, PhysicalDevice>;

}
#endif // DEVICE_H
