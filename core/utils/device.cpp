#include "device.h"
#include "operations.h"

namespace moon::utils {

QueueFamily::QueueFamily(uint32_t index, VkQueueFlags flag, uint32_t queueCount, VkBool32 presentSupport):
index(index), flags(flag), queueCount(queueCount), presentSupport(presentSupport){
    queuePriorities.resize(queueCount, 1.0f/static_cast<float>(queueCount));
}

QueueFamily::QueueFamily(const QueueFamily& other):
index(other.index), flags(other.flags), queueCount(other.queueCount), presentSupport(other.presentSupport){
    queuePriorities.resize(queueCount, 1.0f/static_cast<float>(queueCount));
}

QueueFamily& QueueFamily::operator=(const QueueFamily& other){
    index = other.index;
    flags = other.flags;
    queueCount = other.queueCount;
    presentSupport = other.presentSupport;
    queuePriorities.resize(queueCount, 1.0f/static_cast<float>(queueCount));
    return *this;
}

bool QueueFamily::availableQueueFlag(VkQueueFlags flag) const {
    return (flag & flags) == flag;
}

PhysicalDevice::PhysicalDevice(VkPhysicalDevice physicalDevice, std::vector<const char*> deviceExtensions):
    instance(physicalDevice),
    deviceExtensions(deviceExtensions)
{
    uint32_t queueFamilyPropertyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, queueFamilyProperties.data());

    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
    properties.index = physicalDeviceProperties.deviceID;
    properties.type = physicalDeviceProperties.deviceType;
    properties.name = physicalDeviceProperties.deviceName;

    for (uint32_t index = 0; index < queueFamilyPropertyCount; index++){
        queueFamilies[index] = QueueFamily{index,queueFamilyProperties[index].queueFlags,queueFamilyProperties[index].queueCount, false};
    }
}

bool PhysicalDevice::presentSupport(VkSurfaceKHR surface)
{
    VkBool32 presentSupport = false;
    if(surface){
        for (const auto& [index, _] : queueFamilies){
            VkBool32 support = false;
            CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(instance, index, surface, &support));
            presentSupport |= support;
            queueFamilies[index].presentSupport = support;
        }
    }
    return presentSupport;
}

VkResult PhysicalDevice::createDevice(VkPhysicalDeviceFeatures deviceFeatures, std::map<uint32_t,uint32_t> queueSizeMap)
{
    Device logical(deviceFeatures);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    for(auto queueSize: queueSizeMap){
        if(uint32_t index = queueSize.first; queueFamilies.count(index)){
            queueCreateInfos.push_back(VkDeviceQueueCreateInfo{});
            queueCreateInfos.back().sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfos.back().queueFamilyIndex = index;
            queueCreateInfos.back().queueCount = std::min(queueFamilies[index].queueCount, queueSize.second);
            queueCreateInfos.back().pQueuePriorities = queueFamilies[index].queuePriorities.data();
        }
    }

    VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        createInfo.enabledLayerCount = enableValidationLayers ? static_cast<uint32_t>(validationLayers.size()) : 0;
        createInfo.ppEnabledLayerNames = enableValidationLayers ? validationLayers.data() : nullptr;
    VkResult result = vkCreateDevice(instance, &createInfo, nullptr, &logical.instance);
    CHECK(result);

    for(const auto& queueCreateInfo: queueCreateInfos){
        logical.queueMap[queueCreateInfo.queueFamilyIndex] = std::vector<VkQueue>(queueCreateInfo.queueCount);
        for(uint32_t index = 0; index < queueCreateInfo.queueCount; index++){
            vkGetDeviceQueue(logical.instance, queueCreateInfo.queueFamilyIndex, index, &logical.queueMap[queueCreateInfo.queueFamilyIndex][index]);
        }
    }

    this->logical.emplace_back(std::move(logical));
    return result;
}

VkDevice& PhysicalDevice::getLogical(){
    return logical.back().instance;
}

const VkDevice& PhysicalDevice::getLogical() const{
    return logical.back().instance;
}

bool PhysicalDevice::createdLogical() const {
    return !logical.empty();
}

VkQueue PhysicalDevice::getQueue(uint32_t familyIndex, uint32_t queueIndex) const {
    return logical.back().queueMap.at(familyIndex)[queueIndex];
}

}
