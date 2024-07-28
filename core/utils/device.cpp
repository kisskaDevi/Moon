#include "device.h"
#include "operations.h"

namespace moon::utils {

PhysicalDevice& PhysicalDevice::operator=(PhysicalDevice&& other) {
    swap(other);
    return *this;
};

PhysicalDevice::PhysicalDevice(PhysicalDevice&& other) {
    swap(other);
};

void PhysicalDevice::swap(PhysicalDevice& other) {
    std::swap(descriptor, other.descriptor);
    std::swap(props, other.props);
    std::swap(queueFamilies, other.queueFamilies);
    std::swap(devices, other.devices);
}

PhysicalDevice::PhysicalDevice(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures deviceFeatures, const std::vector<std::string>& deviceExtensions)
    : descriptor(physicalDevice)
{
    uint32_t queueFamilyPropertyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, queueFamilyProperties.data());

    for (uint32_t index = 0; index < queueFamilyPropertyCount; index++) {
        queueFamilies[index] = QueueFamily(queueFamilyProperties[index]);
    }

    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
    props.index = physicalDeviceProperties.deviceID;
    props.type = physicalDeviceProperties.deviceType;
    props.deviceFeatures = deviceFeatures;
    props.name = physicalDeviceProperties.deviceName;
    props.deviceExtensions = deviceExtensions;
}

bool PhysicalDevice::presentSupport(VkSurfaceKHR surface)
{
    VkBool32 presentSupport = false;
    if(surface){
        for (auto& [index, family] : queueFamilies){
            VkBool32 support = false;
            CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(descriptor, index, surface, &support));
            presentSupport |= support;
            family.presentSupport = support;
        }
    }
    return presentSupport;
}

VkResult PhysicalDevice::createDevice(const QueueRequest& queueRequest)
{
    devices.emplace_back(descriptor, props, queueFamilies, queueRequest);
    return VK_SUCCESS;
}

PhysicalDevice::operator VkPhysicalDevice() const {
    return descriptor;
}

const vkDefault::Device& PhysicalDevice::device(uint32_t index) const {
    return devices.at(index);
}

const PhysicalDeviceProperties& PhysicalDevice::properties() const {
    return props;
}

}
