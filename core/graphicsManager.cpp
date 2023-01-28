#include "graphicsManager.h"

#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>

#include <iostream>
#include <set>

graphicsManager::graphicsManager()
{}

void graphicsManager::graphicsManager::createInstance()
{
    VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Graphics Manager";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

    auto extensions = getRequiredExtensions();

    VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        if (enableValidationLayers)
        {
            VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }
    vkCreateInstance(&createInfo, nullptr, &instance);
}
    std::vector<const char*> graphicsManager::getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if(enableValidationLayers)
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        return extensions;
    }
    bool graphicsManager::checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers){
            bool layerFound = false;
            for(const auto& layerProperties: availableLayers){
                if(strcmp(layerName, layerProperties.layerName)==0){
                    layerFound = true;  break;
                }
            }
            if(!layerFound)
                return false;
        }

        return true;
    }

void graphicsManager::graphicsManager::setupDebugMessenger()
{
    if (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger);
}
    void graphicsManager::graphicsManager::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }
        VKAPI_ATTR VkBool32 VKAPI_CALL graphicsManager::debugCallback(
         VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
         VkDebugUtilsMessageTypeFlagsEXT messageType,
         const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
         void* pUserData)
        {
            static_cast<void>(messageSeverity);
            static_cast<void>(messageType);
            static_cast<void>(pUserData);
            std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

            return VK_FALSE;
        }
    VkResult graphicsManager::CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
    {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr)    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
        else                    return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

void graphicsManager::createSurface(GLFWwindow* window)
{
    glfwCreateWindowSurface(instance, window, nullptr, &surface);
}

void graphicsManager::graphicsManager::pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& device : devices)
    {
        std::vector<QueueFamilyIndices> indices = PhysicalDevice::findQueueFamilies(device, surface);
        if (indices.size()!=0 && PhysicalDevice::isSuitable(device,surface,deviceExtensions))
        {
            physicalDevice currentDevice = {device,indices};
            physicalDevices.push_back(currentDevice);
        }
    }

    if(physicalDevices.size()!=0)
    {
        physicalDeviceNumber = 1;
        indicesNumber = 0;
    }
}

void graphicsManager::createLogicalDevice()
{
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).graphicsFamily.value(), physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).presentFamily.value()};

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        deviceFeatures.independentBlend = VK_TRUE;
        deviceFeatures.sampleRateShading = VK_TRUE;
        deviceFeatures.imageCubeArray = VK_TRUE;
        deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else
            createInfo.enabledLayerCount = 0;
    vkCreateDevice(physicalDevices.at(physicalDeviceNumber).device, &createInfo, nullptr, &device);

    vkGetDeviceQueue(device, physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).presentFamily.value(), 0, &presentQueue);
}

void graphicsManager::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).graphicsFamily.value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
}

void graphicsManager::setGraphics(graphicsInterface* graphics)
{
    this->graphics = graphics;

    std::vector<deviceInfo> info;
    info.push_back(deviceInfo{ &physicalDevices.at(physicalDeviceNumber).device,
                               &physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).graphicsFamily,
                               &physicalDevices.at(physicalDeviceNumber).indices.at(indicesNumber).presentFamily,
                               &device,
                               &graphicsQueue,
                               &commandPool});

    this->graphics->setDevicesInfo(static_cast<uint32_t>(info.size()),info.data());
    this->graphics->setSupportImageCount(&surface);
}

void graphicsManager::createGraphics(GLFWwindow* window)
{
    graphics->createGraphics(window,&surface);
}

void graphicsManager::createCommandBuffers()
{
    graphics->createCommandBuffers();
    graphics->updateCommandBuffers();
}

void graphicsManager::createSyncObjects()
{
    VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    vkCreateSemaphore(device, &semaphoreInfo, nullptr, &availableSemaphores);

    signalSemaphores.resize(graphics->getImageCount());
    fences.resize(graphics->getImageCount());

    for (size_t imageIndex = 0; imageIndex < graphics->getImageCount(); imageIndex++){
        VkSemaphoreCreateInfo semaphoreInfo{};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        vkCreateSemaphore(device, &semaphoreInfo, nullptr, &signalSemaphores[imageIndex]);

        VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device, &fenceInfo, nullptr, &fences[imageIndex]);
    }
}

VkResult graphicsManager::checkNextFrame()
{
    VkResult result = vkAcquireNextImageKHR(device, graphics->getSwapChain(), UINT64_MAX, availableSemaphores, VK_NULL_HANDLE, &imageIndex);

    if (result != VK_ERROR_OUT_OF_DATE_KHR)
        vkWaitForFences(device, 1, &fences[imageIndex], VK_TRUE, UINT64_MAX);

    return result;
}

VkResult graphicsManager::drawFrame()
{
    graphics->updateCommandBuffer(imageIndex);
    graphics->updateBuffers(imageIndex);

    VkSemaphore                         waitSemaphores[] = {availableSemaphores};
    VkPipelineStageFlags                waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore                         signalSemaphores[] = {this->signalSemaphores[imageIndex]};

    uint32_t commandBufferCount;
    VkCommandBuffer* commandBufferSet = graphics->getCommandBuffers(commandBufferCount,imageIndex);

    vkResetFences(device, 1, &fences[imageIndex]);
    VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = commandBufferCount;
        submitInfo.pCommandBuffers = commandBufferSet;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, fences[imageIndex]);

    VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &graphics->getSwapChain();
        presentInfo.pImageIndices = &imageIndex;
    return vkQueuePresentKHR(presentQueue, &presentInfo);
}

void graphicsManager::graphicsManager::cleanup()
{
    graphics->freeCommandBuffers();

    if(availableSemaphores) {vkDestroySemaphore(device, availableSemaphores, nullptr); availableSemaphores = VK_NULL_HANDLE;}
    for (size_t imageIndex = 0; imageIndex < graphics->getImageCount(); imageIndex++){
        vkDestroySemaphore(device, signalSemaphores[imageIndex], nullptr);
        vkDestroyFence(device, fences[imageIndex], nullptr);
    }
    signalSemaphores.resize(0);
    fences.resize(0);

    if(commandPool) {vkDestroyCommandPool(device, commandPool, nullptr); commandPool = VK_NULL_HANDLE;}

    if(device) {vkDestroyDevice(device, nullptr); device = VK_NULL_HANDLE;}

    if (enableValidationLayers)
        if(debugMessenger) {DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr); debugMessenger = VK_NULL_HANDLE;}

    if(surface) {vkDestroySurfaceKHR(instance, surface, nullptr); surface = VK_NULL_HANDLE;}
    if(instance) {vkDestroyInstance(instance, nullptr); instance = VK_NULL_HANDLE;}
}
    void graphicsManager::DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
    {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if(func != nullptr)    func(instance, debugMessenger, pAllocator);
    }

uint32_t graphicsManager::getImageIndex(){return imageIndex;}
void     graphicsManager::deviceWaitIdle(){vkDeviceWaitIdle(device);}
