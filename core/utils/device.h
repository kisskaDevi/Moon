#ifndef DEVICE_H
#define DEVICE_H

#include <vulkan.h>
#include <string>
#include <vector>
#include <map>

#include <vkdefault.h>

namespace moon::utils {

class PhysicalDevice{
private:
    VkPhysicalDevice descriptor{VK_NULL_HANDLE};
    PhysicalDeviceProperties props{};
    QueueFamilies queueFamilies;
    vkDefault::Devices devices;

public:
    PhysicalDevice() = default;
    PhysicalDevice(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures deviceFeatures = {}, const std::vector<std::string>& deviceExtensions = {});
    PhysicalDevice& operator=(const PhysicalDevice& other) = delete;
    PhysicalDevice(const PhysicalDevice& other) = delete;
    PhysicalDevice& operator=(PhysicalDevice&& other);
    PhysicalDevice(PhysicalDevice&& other);
    void swap(PhysicalDevice& other);

    VkResult createDevice(const QueueRequest& queueRequest);
    bool presentSupport(VkSurfaceKHR surface);

    operator VkPhysicalDevice() const;
    const vkDefault::Device& device(uint32_t index = 0) const;
    const PhysicalDeviceProperties& properties() const;
};

using PhysicalDeviceMap = std::map<DeviceIndex, PhysicalDevice>;

}
#endif // DEVICE_H
