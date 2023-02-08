#include "device.h"
#include "operations.h"

physicalDevice::physicalDevice(VkPhysicalDevice physicalDevice) : instance(physicalDevice)
{
    PhysicalDevice::printQueueIndices(instance, VK_NULL_HANDLE);

    graphicsQueueIndices = PhysicalDevice::findQueueFamilies(instance, VK_QUEUE_GRAPHICS_BIT);
    computeQueueIndices = PhysicalDevice::findQueueFamilies(instance, VK_QUEUE_COMPUTE_BIT);
    transferQueueIndices = PhysicalDevice::findQueueFamilies(instance, VK_QUEUE_TRANSFER_BIT);
}

physicalDevice::physicalDevice(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) : instance(physicalDevice)
{
    PhysicalDevice::printQueueIndices(instance, surface);

    graphicsQueueIndices = PhysicalDevice::findQueueFamilies(instance, VK_QUEUE_GRAPHICS_BIT, surface);
    computeQueueIndices = PhysicalDevice::findQueueFamilies(instance, VK_QUEUE_COMPUTE_BIT);
    transferQueueIndices = PhysicalDevice::findQueueFamilies(instance, VK_QUEUE_TRANSFER_BIT);
    presentQueueIndices = PhysicalDevice::findQueueFamilies(instance, surface);


}
