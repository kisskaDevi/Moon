#include "graphicsManager.h"

#include <glfw3.h>

#include <string>
#ifndef NDEBUG
#include <iostream>
#endif

graphicsManager::graphicsManager()
{
    debug::checkResult(createInstance(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
}

VkResult graphicsManager::createInstance()
{
    VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Graphics Manager";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

    enableValidationLayers &= ValidationLayer::checkSupport(validationLayers);

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    if(enableValidationLayers){
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCreateInfo.pfnUserCallback = ValidationLayer::debugCallback;
    VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        createInfo.enabledLayerCount = enableValidationLayers ? static_cast<uint32_t>(validationLayers.size()) : 0;
        createInfo.ppEnabledLayerNames = enableValidationLayers ? validationLayers.data() : nullptr;
        createInfo.pNext = enableValidationLayers ? (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo : nullptr;
    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

    if (enableValidationLayers){
        ValidationLayer::setupDebugMessenger(instance, &debugMessenger);
    }

    return result;
}

VkResult graphicsManager::createDevice()
{
    if(surface == VK_NULL_HANDLE)   return debug::errorResult("[ createDevice ] surface is VK_NULL_HANDLE");
    if(instance == VK_NULL_HANDLE)  return debug::errorResult("[ createDevice ] instance is VK_NULL_HANDLE");

    VkResult result = VK_SUCCESS;

    uint32_t deviceCount = 0;
    result = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    debug::checkResult(result, "VkInstance : vkEnumeratePhysicalDevices result = " + std::to_string(result));

    std::vector<VkPhysicalDevice> phDevices(deviceCount);
    result = vkEnumeratePhysicalDevices(instance, &deviceCount, phDevices.data());
    debug::checkResult(result, "VkInstance : vkEnumeratePhysicalDevices result = " + std::to_string(result));

    for (const auto device : phDevices){
        devices.emplace_back(physicalDevice(device,surface,deviceExtensions));
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        deviceFeatures.independentBlend = VK_TRUE;
        deviceFeatures.sampleRateShading = VK_TRUE;
        deviceFeatures.imageCubeArray = VK_TRUE;
        deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;

    device logical(deviceFeatures);
    result = devices[0].createDevice(logical,{{0,2}});
    return result;
}

VkResult graphicsManager::createSurface(GLFWwindow* window)
{
    if(instance == VK_NULL_HANDLE)  return debug::errorResult("[ createSurface ] instance is VK_NULL_HANDLE");
    return window ? glfwCreateWindowSurface(instance, window, nullptr, &surface) : debug::errorResult("[ createSurface ] Window is nullptr");
}

VkResult graphicsManager::createSwapChain(GLFWwindow* window, int32_t maxImageCount)
{
    if(window == nullptr)           return debug::errorResult("[ createSwapChain ] Window is nullptr");
    if(surface == VK_NULL_HANDLE)   return debug::errorResult("[ createSwapChain ] surface is VK_NULL_HANDLE");
    if(devices.empty())   return debug::errorResult("[ createSyncObjects ] device is VK_NULL_HANDLE");

    std::vector<uint32_t> queueIndices = {0};
    swapChainKHR.setDevice(devices[0].instance, devices[0].getLogical());
    return swapChainKHR.create(window, &surface, static_cast<uint32_t>(queueIndices.size()), queueIndices.data(), maxImageCount);
}

void graphicsManager::setGraphics(graphicsInterface* graphics)
{
    this->graphics.push_back(graphics);
    this->graphics.back()->setDevices(static_cast<uint32_t>(devices.size()), devices.data());
    this->graphics.back()->setSwapChain(&swapChainKHR);
    this->graphics.back()->setImageCount(swapChainKHR.getImageCount());
}

VkResult graphicsManager::createSyncObjects()
{
    if(devices.empty())   return debug::errorResult("[ createSyncObjects ] device is VK_NULL_HANDLE");

    VkResult result = VK_SUCCESS;

    availableSemaphores.resize(getImageCount());
    signalSemaphores.resize(getImageCount());
    fences.resize(getImageCount());

    for (size_t imageIndex = 0; imageIndex < getImageCount(); imageIndex++){
        VkSemaphoreCreateInfo semaphoreInfo{};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        vkCreateSemaphore(devices[0].getLogical(), &semaphoreInfo, nullptr, &signalSemaphores[imageIndex]);
        vkCreateSemaphore(devices[0].getLogical(), &semaphoreInfo, nullptr, &availableSemaphores[imageIndex]);

        VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        result = vkCreateFence(devices[0].getLogical(), &fenceInfo, nullptr, &fences[imageIndex]);
        debug::checkResult(result, "VkInstance : vkEnumeratePhysicalDevices result = " + std::to_string(result));
    }
    return result;
}

VkResult graphicsManager::checkNextFrame()
{
    VkResult result = vkAcquireNextImageKHR(devices[0].getLogical(), swapChainKHR(), UINT64_MAX, availableSemaphores[semaphorIndex], VK_NULL_HANDLE, &imageIndex);

    if (result != VK_ERROR_OUT_OF_DATE_KHR)
        result = vkWaitForFences(devices[0].getLogical(), 1, &fences[imageIndex], VK_TRUE, UINT64_MAX);

    return result;
}

VkResult graphicsManager::drawFrame()
{
    for(auto graphics: graphics){
        graphics->updateBuffers(imageIndex);
        graphics->updateCommandBuffer(imageIndex);
    }

    vkResetFences(devices[0].getLogical(), 1, &fences[imageIndex]);

    std::vector<std::vector<VkFence>> signalFences(graphics.size() - 1);
    std::vector<std::vector<VkSemaphore>> waitSemaphores = {{availableSemaphores[semaphorIndex]}};
    for(size_t i = 0; i < graphics.size() - 1; i++){
        waitSemaphores = graphics[i]->sibmit(waitSemaphores,signalFences[i],imageIndex);
    }
    signalFences.push_back({fences[imageIndex]});
    waitSemaphores = graphics.back()->sibmit(waitSemaphores,signalFences.back(),imageIndex);

    semaphorIndex = ((semaphorIndex + 1) % getImageCount());

    VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.back().size());
        presentInfo.pWaitSemaphores = waitSemaphores.back().data();
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapChainKHR();
        presentInfo.pImageIndices = &imageIndex;
    return vkQueuePresentKHR(devices[0].getQueue(0,0), &presentInfo);
}

void graphicsManager::destroySwapChain()
{
    swapChainKHR.destroy();
}

void graphicsManager::destroy()
{
    for (size_t imageIndex = 0; imageIndex < getImageCount(); imageIndex++){
        vkDestroySemaphore(devices[0].getLogical(), availableSemaphores[imageIndex], nullptr);
        vkDestroySemaphore(devices[0].getLogical(), signalSemaphores[imageIndex], nullptr);
        vkDestroyFence(devices[0].getLogical(), fences[imageIndex], nullptr);
    }
    availableSemaphores.resize(0);
    signalSemaphores.resize(0);
    fences.resize(0);

    if(devices[0].getLogical())                     {vkDestroyDevice(devices[0].getLogical(), nullptr); devices[0].getLogical() = VK_NULL_HANDLE;}
    if(enableValidationLayers && debugMessenger)    { ValidationLayer::DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr); debugMessenger = VK_NULL_HANDLE;}
    if(surface)                                     {vkDestroySurfaceKHR(instance, surface, nullptr); surface = VK_NULL_HANDLE;}
    if(instance)                                    {vkDestroyInstance(instance, nullptr); instance = VK_NULL_HANDLE;}
}

VkSurfaceKHR&   graphicsManager::getSurface(){return surface;}
uint32_t        graphicsManager::getImageIndex(){return imageIndex;}
uint32_t        graphicsManager::getImageCount(){return swapChainKHR.getImageCount();}
VkResult        graphicsManager::deviceWaitIdle(){return vkDeviceWaitIdle(devices[0].getLogical());}
