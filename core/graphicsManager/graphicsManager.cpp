#include "graphicsManager.h"

#include <glfw3.h>

#include "linkable.h"

#include <string>

graphicsManager::graphicsManager(const VkPhysicalDeviceFeatures& deviceFeatures, size_t deviceIndex) : deviceIndex(deviceIndex)
{
    debug::checkResult(createInstance(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(createDevice(deviceFeatures), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
}

graphicsManager::graphicsManager(GLFWwindow* window, int32_t maxImageCount, const VkPhysicalDeviceFeatures& deviceFeatures, size_t deviceIndex)
    : graphicsManager(deviceFeatures, deviceIndex)
{
    create(window, maxImageCount);
}

graphicsManager::~graphicsManager(){
    destroy();

    if(devices[deviceIndex].getLogical())           { vkDestroyDevice(devices[deviceIndex].getLogical(), nullptr); devices[deviceIndex].getLogical() = VK_NULL_HANDLE;}
    if(enableValidationLayers && debugMessenger)    { ValidationLayer::DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr); debugMessenger = VK_NULL_HANDLE;}
    if(surface)                                     { vkDestroySurfaceKHR(instance, surface, nullptr); surface = VK_NULL_HANDLE;}
    if(instance)                                    { vkDestroyInstance(instance, nullptr); instance = VK_NULL_HANDLE;}
}

void graphicsManager::create(GLFWwindow* window, int32_t maxImageCount){
    debug::checkResult(createSurface(window), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(createSwapChain(window, maxImageCount), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(createLinker(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    debug::checkResult(createSyncObjects(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
}

VkResult graphicsManager::createInstance(){
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
    CHECK(result)

    if (enableValidationLayers){
        ValidationLayer::setupDebugMessenger(instance, &debugMessenger);
    }

    return result;
}

VkResult graphicsManager::createDevice(const VkPhysicalDeviceFeatures& deviceFeatures){
    CHECK_M(instance == VK_NULL_HANDLE, "[ graphicsManager::createDevice ] instance is VK_NULL_HANDLE");

    VkResult result = VK_SUCCESS;

    uint32_t deviceCount = 0;
    result = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    CHECK(result);

    std::vector<VkPhysicalDevice> phDevices(deviceCount);
    result = vkEnumeratePhysicalDevices(instance, &deviceCount, phDevices.data());
    CHECK(result);

    for (const auto device : phDevices){
        devices.emplace_back(physicalDevice(device, deviceExtensions));
    }

    device logical(deviceFeatures);
    result = devices[deviceIndex].createDevice(logical,{{0,2}});
    CHECK(result);

    return result;
}

VkResult graphicsManager::createSurface(GLFWwindow* window){
    CHECK_M(instance == VK_NULL_HANDLE, "[ graphicsManager::createSurface ] instance is VK_NULL_HANDLE");
    CHECK_M(devices.empty(), "[ graphicsManager::createSwapChain ] device is VK_NULL_HANDLE");
    CHECK_M(window == nullptr, "[ createSurface ] Window is nullptr");

    VkResult result = glfwCreateWindowSurface(instance, window, nullptr, &surface);
    CHECK(result);
    CHECK_M(!devices[deviceIndex].presentSupport(surface), "[ graphicsManager::createSurface ] device doesn't support present");

    return result;
}

VkResult graphicsManager::createSwapChain(GLFWwindow* window, int32_t maxImageCount){
    CHECK_M(window == nullptr, "[ graphicsManager::createSwapChain ] Window is nullptr");
    CHECK_M(surface == VK_NULL_HANDLE, "[ graphicsManager::createSwapChain ] surface is VK_NULL_HANDLE");
    CHECK_M(devices.empty(), "[ graphicsManager::createSwapChain ] device is VK_NULL_HANDLE");

    std::vector<uint32_t> queueIndices = {0};
    swapChainKHR.setDevice(devices[deviceIndex]);
    return swapChainKHR.create(window, surface, static_cast<uint32_t>(queueIndices.size()), queueIndices.data(), maxImageCount);
}

VkResult graphicsManager::createLinker(){
    linker.setDevice(devices[deviceIndex].getLogical());
    linker.setSwapChain(&swapChainKHR);
    linker.createRenderPass();
    linker.createFramebuffers();
    linker.createCommandPool();
    linker.createCommandBuffers();
    linker.createSyncObjects();

    for(auto graphics: graphics){
        graphics->getLinkable()->setRenderPass(linker.getRenderPass());
    }

    return VK_SUCCESS;
}

void graphicsManager::setGraphics(graphicsInterface* graphics){
    this->graphics.push_back(graphics);
    this->graphics.back()->setDevices(static_cast<uint32_t>(devices.size()), devices.data(), deviceIndex);
    this->graphics.back()->setSwapChain(&swapChainKHR);
    this->graphics.back()->getLinkable()->setRenderPass(linker.getRenderPass());
    linker.addLinkable(this->graphics.back()->getLinkable());
}

VkResult graphicsManager::createSyncObjects(){
    CHECK_M(devices.empty(), "[ graphicsManager::createSyncObjects ] device is VK_NULL_HANDLE");

    VkResult result = VK_SUCCESS;

    availableSemaphores.resize(swapChainKHR.getImageCount());
    fences.resize(swapChainKHR.getImageCount());

    for (size_t imageIndex = 0; imageIndex < swapChainKHR.getImageCount(); imageIndex++){
        VkSemaphoreCreateInfo semaphoreInfo{};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        result = vkCreateSemaphore(devices[deviceIndex].getLogical(), &semaphoreInfo, nullptr, &availableSemaphores[imageIndex]);
        CHECK(result);

        VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        result = vkCreateFence(devices[deviceIndex].getLogical(), &fenceInfo, nullptr, &fences[imageIndex]);
        CHECK(result);
    }
    return result;
}

VkResult graphicsManager::checkNextFrame(){
    VkResult result = vkWaitForFences(devices[deviceIndex].getLogical(), 1, &fences[resourceIndex], VK_TRUE, UINT64_MAX);
    CHECK(result);

    result = vkResetFences(devices[deviceIndex].getLogical(), 1, &fences[resourceIndex]);
    CHECK(result);

    result = vkAcquireNextImageKHR(devices[deviceIndex].getLogical(), swapChainKHR(), UINT64_MAX, availableSemaphores[resourceIndex], VK_NULL_HANDLE, &imageIndex);
    CHECK(result);

    return result;
}

VkResult graphicsManager::drawFrame(){
    for(auto graphics: graphics){
        graphics->update(resourceIndex);
    }
    linker.updateCmdFlags();
    linker.updateCommandBuffer(resourceIndex, imageIndex);

    std::vector<std::vector<VkSemaphore>> waitSemaphores = {{availableSemaphores[resourceIndex]}};
    for(auto& graph: graphics){
        waitSemaphores = graph->submit(waitSemaphores, {VK_NULL_HANDLE}, resourceIndex);
    }

    VkSemaphore linkerSemaphore = linker.submit(
        resourceIndex,
        waitSemaphores.size() > 0 ? waitSemaphores.back() : std::vector<VkSemaphore>(),
        fences[resourceIndex],
        devices[deviceIndex].getQueue(0,0)
    );

    resourceIndex = ((resourceIndex + 1) % swapChainKHR.getImageCount());

    VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &linkerSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapChainKHR();
        presentInfo.pImageIndices = &imageIndex;
    return vkQueuePresentKHR(devices[deviceIndex].getQueue(0,0), &presentInfo);
}

void graphicsManager::destroySwapChain(){
    swapChainKHR.destroy();
}

void graphicsManager::destroyLinker(){
    linker.destroy();
}

void graphicsManager::destroySyncObjects(){
    for (auto& semaphore : availableSemaphores){
        vkDestroySemaphore(devices[deviceIndex].getLogical(), semaphore, nullptr);
    }
    availableSemaphores.clear();
    for (auto& fence : fences){
        vkDestroyFence(devices[deviceIndex].getLogical(), fence, nullptr);
    }
    fences.clear();
}

void graphicsManager::destroySurface(){
    if(surface) {
        vkDestroySurfaceKHR(instance, surface, nullptr); surface = VK_NULL_HANDLE;
    }
}

void graphicsManager::destroy(){
    destroySwapChain();
    destroyLinker();
    destroySyncObjects();
    destroySurface();
}

VkInstance      graphicsManager::getInstance()      {return instance;}
VkSurfaceKHR    graphicsManager::getSurface()       {return surface;}
uint32_t        graphicsManager::getImageIndex()    {return imageIndex;}
swapChain*      graphicsManager::getSwapChain()     {return &swapChainKHR;}
VkResult        graphicsManager::deviceWaitIdle()   {return vkDeviceWaitIdle(devices[deviceIndex].getLogical());}
