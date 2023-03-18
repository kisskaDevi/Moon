#ifndef DEVICE_H
#define DEVICE_H

#include <vulkan.h>

#include <vector>
#include <map>
#include <algorithm>

struct queueFamily{
    uint32_t index;
    VkQueueFlags flags;
    uint32_t queueCount;
    VkBool32 presentSupport;
    std::vector<float> queuePriorities;

    queueFamily() = default;

    queueFamily(uint32_t index, VkQueueFlags flag, uint32_t queueCount, VkBool32 presentSupport):
    index(index), flags(flag), queueCount(queueCount), presentSupport(presentSupport){
        queuePriorities.resize(queueCount, 1.0f/static_cast<float>(queueCount));
    }

    queueFamily(const queueFamily& other):
    index(other.index), flags(other.flags), queueCount(other.queueCount), presentSupport(other.presentSupport){
        queuePriorities.resize(queueCount, 1.0f/static_cast<float>(queueCount));
    }

    queueFamily& operator=(const queueFamily& other){
        index = other.index;
        flags = other.flags;
        queueCount = other.queueCount;
        presentSupport = other.presentSupport;
        queuePriorities.resize(queueCount, 1.0f/static_cast<float>(queueCount));
        return *this;
    }

    bool availableQueueFlag(VkQueueFlags flag){
        return (flag & flags) == flag;
    }
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
    physicalDevice(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface = VK_NULL_HANDLE, std::vector<const char*> deviceExtensions = {}):
        instance(physicalDevice),
        deviceExtensions(deviceExtensions)
    {
        uint32_t queueFamilyPropertyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, nullptr);

        std::vector<VkQueueFamilyProperties> aueueFamilyProperties(queueFamilyPropertyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, aueueFamilyProperties.data());

        for (uint32_t index = 0; index < queueFamilyPropertyCount; index++){
            VkBool32 presentSupport = surface ? false : true;
            if(surface){
                vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, index, surface, &presentSupport);
            }
            queueFamilies[index] = queueFamily{index,aueueFamilyProperties[index].queueFlags,aueueFamilyProperties[index].queueCount, presentSupport};
        }
    }

    void createDevice(device logical, std::map<uint32_t,uint32_t> queueSizeMap)
    {
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
            createInfo.pEnabledFeatures = &logical.deviceFeatures;
            createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
            createInfo.ppEnabledExtensionNames = deviceExtensions.data();
            createInfo.enabledLayerCount = enableValidationLayers ? static_cast<uint32_t>(validationLayers.size()) : 0;
            createInfo.ppEnabledLayerNames = enableValidationLayers ? validationLayers.data() : nullptr;
        vkCreateDevice(instance, &createInfo, nullptr, &logical.instance);

        for(auto queueCreateInfo: queueCreateInfos){
            logical.queueMap[queueCreateInfo.queueFamilyIndex] = std::vector<VkQueue>(queueCreateInfo.queueCount);
            for(uint32_t index = 0; index < queueCreateInfo.queueCount; index++){
                vkGetDeviceQueue(logical.instance, queueCreateInfo.queueFamilyIndex, index, &logical.queueMap[queueCreateInfo.queueFamilyIndex][index]);
            }
        }

        this->logical.emplace_back(logical);
    }

    VkDevice& getLogical(){return logical.back().instance;}

    physicalDevice& operator=(const physicalDevice& other){
        instance = other.instance;
        queueFamilies = other.queueFamilies;
        logical = other.logical;
        deviceExtensions = other.deviceExtensions;
        return *this;
    }

    physicalDevice(const physicalDevice& other):
        instance(other.instance),
        queueFamilies(other.queueFamilies),
        logical(other.logical),
        deviceExtensions(other.deviceExtensions)
    {}

    VkQueue getQueue(uint32_t familyIndex, uint32_t queueIndex){
        return logical.back().queueMap[familyIndex][queueIndex];
    }
};

#endif // DEVICE_H
