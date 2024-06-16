#include "graphicsManager.h"
#include "linkable.h"

#include <glfw3.h>
#include <string>

namespace moon::graphicsManager {

GraphicsManager::GraphicsManager(const VkPhysicalDeviceFeatures& deviceFeatures){
    moon::utils::debug::checkResult(createInstance(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    moon::utils::debug::checkResult(createDevice(deviceFeatures), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
}

GraphicsManager::GraphicsManager(GLFWwindow* window, int32_t imageCount, int32_t resourceCount, const VkPhysicalDeviceFeatures& deviceFeatures)
    : GraphicsManager(deviceFeatures){
    this->imageCount = imageCount;
    this->resourceCount = resourceCount;
    create(window);
}

GraphicsManager::~GraphicsManager(){
    destroy();
}

void GraphicsManager::create(GLFWwindow* window){
    moon::utils::debug::checkResult(createSurface(window), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    moon::utils::debug::checkResult(createSwapChain(window, imageCount), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    moon::utils::debug::checkResult(createLinker(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    moon::utils::debug::checkResult(createSyncObjects(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
}

VkResult GraphicsManager::createInstance(){
    VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Graphics Manager";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

    enableValidationLayers &= moon::utils::validationLayer::checkSupport(validationLayers);

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
        debugCreateInfo.pfnUserCallback = moon::utils::validationLayer::debugCallback;
    VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        createInfo.enabledLayerCount = enableValidationLayers ? static_cast<uint32_t>(validationLayers.size()) : 0;
        createInfo.ppEnabledLayerNames = enableValidationLayers ? validationLayers.data() : nullptr;
        createInfo.pNext = enableValidationLayers ? (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo : nullptr;
    VkResult result = instance.create(createInfo);
    CHECK(result)

    if (enableValidationLayers) debugMessenger.create(instance);

    return result;
}

VkResult GraphicsManager::createDevice(const VkPhysicalDeviceFeatures& deviceFeatures){
    CHECK_M(instance == VK_NULL_HANDLE, "[ GraphicsManager::createDevice ] instance is VK_NULL_HANDLE");

    VkResult result = VK_SUCCESS;

    uint32_t deviceCount = 0;
    result = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    CHECK(result);

    std::vector<VkPhysicalDevice> phDevices(deviceCount);
    result = vkEnumeratePhysicalDevices(instance, &deviceCount, phDevices.data());
    CHECK(result);

    for (const auto phDevice : phDevices){
        auto device = moon::utils::PhysicalDevice(phDevice, deviceExtensions);
        const moon::utils::DeviceIndex index = device.properties.index;
        devices[index] = std::move(device);
        if(!activeDevice){
            activeDevice = &devices[index];
        }
        if(activeDevice->properties.type != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
            devices[index].properties.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU){
            activeDevice = &devices[index];
        }
    }

    if(!activeDevice){
        return VK_ERROR_DEVICE_LOST;
    }

    result = activeDevice->createDevice(deviceFeatures, {{0,2}});
    CHECK(result);

    return result;
}

VkResult GraphicsManager::createSurface(GLFWwindow* window){
    CHECK_M(instance == VK_NULL_HANDLE, "[ GraphicsManager::createSurface ] instance is VK_NULL_HANDLE");
    CHECK_M(devices.empty(), "[ GraphicsManager::createSwapChain ] device is VK_NULL_HANDLE");
    CHECK_M(window == nullptr, "[ createSurface ] Window is nullptr");

    VkResult result = surface.create(instance, window);
    CHECK(result);
    CHECK_M(!activeDevice->presentSupport(surface), "[ GraphicsManager::createSurface ] device doesn't support present");

    return result;
}

VkResult GraphicsManager::createSwapChain(GLFWwindow* window, int32_t maxImageCount){
    CHECK_M(window == nullptr, "[ GraphicsManager::createSwapChain ] Window is nullptr");
    CHECK_M(surface == VK_NULL_HANDLE, "[ GraphicsManager::createSwapChain ] surface is VK_NULL_HANDLE");
    CHECK_M(devices.empty(), "[ GraphicsManager::createSwapChain ] device is VK_NULL_HANDLE");

    std::vector<uint32_t> queueIndices = {0};
    swapChainKHR.setDevice(activeDevice);
    return swapChainKHR.create(window, surface, static_cast<uint32_t>(queueIndices.size()), queueIndices.data(), maxImageCount);
}

VkResult GraphicsManager::createLinker(){
    linker.setDevice(activeDevice->getLogical());
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

void GraphicsManager::setGraphics(GraphicsInterface* graphics){
    this->graphics.push_back(graphics);
    this->graphics.back()->setDevices(devices, activeDevice->properties.index);
    this->graphics.back()->setSwapChain(&swapChainKHR);
    this->graphics.back()->setProperties(swapChainKHR.getFormat(), resourceCount);
    this->graphics.back()->getLinkable()->setRenderPass(linker.getRenderPass());
    linker.addLinkable(this->graphics.back()->getLinkable());
}

std::vector<moon::utils::PhysicalDeviceProperties> GraphicsManager::getDeviceInfo() const {
    std::vector<moon::utils::PhysicalDeviceProperties> deviceProperties;
    for(const auto& [_,device]: devices){
        deviceProperties.push_back(device.properties);
    }
    return deviceProperties;
}

void GraphicsManager::setDevice(uint32_t deviceIndex){
    activeDevice = &devices.at(deviceIndex);
}

VkResult GraphicsManager::createSyncObjects(){
    CHECK_M(devices.empty(), "[ GraphicsManager::createSyncObjects ] device is VK_NULL_HANDLE");

    VkResult result = VK_SUCCESS;

    availableSemaphores.resize(resourceCount);
    fences.resize(resourceCount);

    for (size_t imageIndex = 0; imageIndex < resourceCount; imageIndex++){
        VkSemaphoreCreateInfo semaphoreInfo{};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        result = vkCreateSemaphore(activeDevice->getLogical(), &semaphoreInfo, nullptr, &availableSemaphores[imageIndex]);
        CHECK(result);

        VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        result = vkCreateFence(activeDevice->getLogical(), &fenceInfo, nullptr, &fences[imageIndex]);
        CHECK(result);
    }
    return result;
}

VkResult GraphicsManager::checkNextFrame(){
    VkResult result = vkWaitForFences(activeDevice->getLogical(), 1, &fences[resourceIndex], VK_TRUE, UINT64_MAX);
    CHECK(result);

    result = vkResetFences(activeDevice->getLogical(), 1, &fences[resourceIndex]);
    CHECK(result);

    result = vkAcquireNextImageKHR(activeDevice->getLogical(), swapChainKHR(), UINT64_MAX, availableSemaphores[resourceIndex], VK_NULL_HANDLE, &imageIndex);
    CHECK(result);

    return result;
}

VkResult GraphicsManager::drawFrame(){
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
        activeDevice->getQueue(0,0)
    );

    resourceIndex = ((resourceIndex + 1) % resourceCount);

    VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &linkerSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapChainKHR();
        presentInfo.pImageIndices = &imageIndex;
    return vkQueuePresentKHR(activeDevice->getQueue(0,0), &presentInfo);
}

VkResult GraphicsManager::deviceWaitIdle() const {
    return vkDeviceWaitIdle(activeDevice->getLogical());
}

void GraphicsManager::destroySwapChain(){
    swapChainKHR.destroy();
}

void GraphicsManager::destroyLinker(){
    linker.destroy();
}

void GraphicsManager::destroySyncObjects(){
    for (auto& semaphore : availableSemaphores){
        vkDestroySemaphore(activeDevice->getLogical(), semaphore, nullptr);
    }
    availableSemaphores.clear();
    for (auto& fence : fences){
        vkDestroyFence(activeDevice->getLogical(), fence, nullptr);
    }
    fences.clear();
}

void GraphicsManager::destroy(){
    destroySwapChain();
    destroyLinker();
    destroySyncObjects();
}

VkInstance      GraphicsManager::getInstance()      const {return instance;}
VkExtent2D      GraphicsManager::getImageExtent()   const {return swapChainKHR.getExtent();}
uint32_t        GraphicsManager::getResourceIndex() const {return resourceIndex;}
uint32_t        GraphicsManager::getResourceCount() const {return resourceCount;}
uint32_t        GraphicsManager::getImageIndex()    const {return imageIndex;}
uint32_t        GraphicsManager::getImageCount()    const {return imageCount;}

std::vector<uint32_t> GraphicsManager::makeScreenshot() const {
    return swapChainKHR.makeScreenshot(imageIndex);
}

}
