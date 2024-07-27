#include "graphicsManager.h"
#include "linkable.h"

#include <glfw3.h>
#include <string>

namespace moon::graphicsManager {

GraphicsManager::GraphicsManager(GLFWwindow* window, int32_t imageCount, int32_t resourceCount, const VkPhysicalDeviceFeatures& deviceFeatures) :
    imageCount(imageCount),
    resourceCount(resourceCount) {
    moon::utils::debug::checkResult(createInstance(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    moon::utils::debug::checkResult(createDevice(deviceFeatures), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    moon::utils::debug::checkResult(createSurface(window), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    reset(window);
}

void GraphicsManager::reset(GLFWwindow* window){
    deviceWaitIdle();
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
    VkResult result = VK_SUCCESS;
    instance = utils::vkDefault::Instance(createInfo);

    if (enableValidationLayers) debugMessenger = utils::vkDefault::DebugUtilsMessenger(instance);

    return VK_SUCCESS;
}

VkResult GraphicsManager::createDevice(const VkPhysicalDeviceFeatures& deviceFeatures){
    CHECK_M(instance == VK_NULL_HANDLE, "[ GraphicsManager::createDevice ] instance is VK_NULL_HANDLE");

    VkResult result = VK_SUCCESS;

    uint32_t deviceCount = 0;
    CHECK(result = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));

    std::vector<VkPhysicalDevice> phDevices(deviceCount);
    CHECK(result = vkEnumeratePhysicalDevices(instance, &deviceCount, phDevices.data()));

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

    if(!activeDevice) return VK_ERROR_DEVICE_LOST;

    CHECK(result = activeDevice->createDevice(deviceFeatures, { {0,2} }));

    return result;
}

VkResult GraphicsManager::createSurface(GLFWwindow* window){
    CHECK_M(instance == VK_NULL_HANDLE, "[ GraphicsManager::createSurface ] instance is VK_NULL_HANDLE");
    CHECK_M(devices.empty(), "[ GraphicsManager::createSwapChain ] device is VK_NULL_HANDLE");
    CHECK_M(window == nullptr, "[ createSurface ] Window is nullptr");

    surface = utils::vkDefault::Surface(instance, window);
    CHECK_M(!activeDevice->presentSupport(surface), "[ GraphicsManager::createSurface ] device doesn't support present");

    return VK_SUCCESS;
}

VkResult GraphicsManager::createSwapChain(GLFWwindow* window, int32_t maxImageCount){
    CHECK_M(window == nullptr, "[ GraphicsManager::createSwapChain ] Window is nullptr");
    CHECK_M(surface == VK_NULL_HANDLE, "[ GraphicsManager::createSwapChain ] surface is VK_NULL_HANDLE");
    CHECK_M(activeDevice == nullptr, "[ GraphicsManager::activeDevice ] device is nullptr");
    return swapChainKHR.reset(activeDevice, window, surface, maxImageCount);
}

VkResult GraphicsManager::createLinker(){
    linker = GraphicsLinker(activeDevice->getLogical(), &swapChainKHR, &graphics);
    return VK_SUCCESS;
}

void GraphicsManager::setGraphics(GraphicsInterface* ingraphics){
    graphics.push_back(ingraphics);
    ingraphics->setProperties(devices, activeDevice->properties.index, &swapChainKHR, resourceCount);
    ingraphics->link->renderPass() = linker.getRenderPass();
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
    for (auto& semaphore : availableSemaphores) {
        semaphore = utils::vkDefault::Semaphore(activeDevice->getLogical());
    }

    fences.resize(resourceCount);
    for (auto& fence : fences) {
        fence = utils::vkDefault::Fence(activeDevice->getLogical());
    }

    return result;
}

VkResult GraphicsManager::checkNextFrame(){
    VkResult result = VK_SUCCESS;

    CHECK(result = vkWaitForFences(activeDevice->getLogical(), 1, fences[resourceIndex], VK_TRUE, UINT64_MAX));
    CHECK(result = vkResetFences(activeDevice->getLogical(), 1, fences[resourceIndex]));
    CHECK(result = vkAcquireNextImageKHR(activeDevice->getLogical(), swapChainKHR, UINT64_MAX, availableSemaphores[resourceIndex], VK_NULL_HANDLE, &imageIndex));

    return result;
}

VkResult GraphicsManager::drawFrame(){
    for(auto graphics: graphics){
        graphics->update(resourceIndex);
    }
    linker.update(resourceIndex, imageIndex);

    utils::vkDefault::VkSemaphores waitSemaphores = {availableSemaphores[resourceIndex]};
    for(auto& graph: graphics){
        waitSemaphores = graph->submit(resourceIndex, {VK_NULL_HANDLE}, waitSemaphores);
    }

    VkSemaphore linkerSemaphore = linker.submit(resourceIndex, waitSemaphores, fences[resourceIndex], activeDevice->getQueue(0,0));

    resourceIndex = ((resourceIndex + 1) % resourceCount);

    return swapChainKHR.present(linkerSemaphore, imageIndex);
}

VkResult GraphicsManager::deviceWaitIdle() const {
    return vkDeviceWaitIdle(activeDevice->getLogical());
}

VkInstance      GraphicsManager::getInstance()      const {return instance;}
VkExtent2D      GraphicsManager::getImageExtent()   const {return swapChainKHR.info().Extent;}
uint32_t        GraphicsManager::getResourceIndex() const {return resourceIndex;}
uint32_t        GraphicsManager::getResourceCount() const {return resourceCount;}
uint32_t        GraphicsManager::getImageIndex()    const {return imageIndex;}
uint32_t        GraphicsManager::getImageCount()    const {return imageCount;}

std::vector<uint32_t> GraphicsManager::makeScreenshot() const {
    return swapChainKHR.makeScreenshot(imageIndex);
}

}
