#include "swapChain.h"
#include "buffer.h"
#include <glfw3.h>
#include <cstring>

namespace moon::utils {

void SwapChain::destroy(){
    for (auto& instance: swapChainAttachments.instances){
        if(instance.imageView){
            vkDestroyImageView(device->getLogical(), instance.imageView, nullptr);
            instance.imageView = VK_NULL_HANDLE;
        }
    }

    if(swapChainKHR) {
        vkDestroySwapchainKHR(device->getLogical(), swapChainKHR, nullptr);
        swapChainKHR = VK_NULL_HANDLE;
    }

    if(commandPool) {
        vkDestroyCommandPool(device->getLogical(), commandPool, nullptr); commandPool = VK_NULL_HANDLE;
    }
}

VkResult SwapChain::create(GLFWwindow* window, VkSurfaceKHR surface, uint32_t queueFamilyIndexCount, uint32_t* pQueueFamilyIndices, int32_t maxImageCount){
    VkResult result = VK_SUCCESS;

    this->window = window;
    this->surface = surface;

    swapChain::SupportDetails swapChainSupport = swapChain::queryingSupport(device->instance, surface);
    VkSurfaceFormatKHR surfaceFormat = swapChain::queryingSurfaceFormat(swapChainSupport.formats);
    VkSurfaceCapabilitiesKHR capabilities = swapChain::queryingSupport(device->instance, surface).capabilities;

    imageCount = swapChain::queryingSupportImageCount(device->instance, surface);
    imageCount = (maxImageCount > 0 && imageCount > static_cast<uint32_t>(maxImageCount)) ? static_cast<uint32_t>(maxImageCount) : imageCount;
    extent = swapChain::queryingExtent(window, capabilities);
    format = surfaceFormat.format;

    VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        createInfo.imageSharingMode = queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
        createInfo.pQueueFamilyIndices = pQueueFamilyIndices;
        createInfo.queueFamilyIndexCount = queueFamilyIndexCount;
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = swapChain::queryingPresentMode(swapChainSupport.presentModes);
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
    result = vkCreateSwapchainKHR(device->getLogical(), &createInfo, nullptr, &swapChainKHR);
    CHECK(result);

    swapChainAttachments.instances.resize(imageCount);
    std::vector<VkImage> images = swapChainAttachments.getImages();
    result = vkGetSwapchainImagesKHR(device->getLogical(), swapChainKHR, &imageCount, images.data());
    CHECK(result);

    for (auto& instance: swapChainAttachments.instances){
        instance.image = images[&instance - &swapChainAttachments.instances[0]];
        result = texture::createView(   device->getLogical(),
                                        VK_IMAGE_VIEW_TYPE_2D,
                                        surfaceFormat.format,
                                        VK_IMAGE_ASPECT_COLOR_BIT,
                                        1,
                                        0,
                                        1,
                                        instance.image,
                                        &instance.imageView);
        CHECK(result);
    }

    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = 0;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    result = vkCreateCommandPool(device->getLogical(), &poolInfo, nullptr, &commandPool);
    CHECK(result);

    return result;
}

VkSwapchainKHR& SwapChain::operator()(){
    return swapChainKHR;
}

Attachment& SwapChain::attachment(uint32_t i){
    return swapChainAttachments.instances[i];
}

void SwapChain::setDevice(const const PhysicalDevice* device){
    this->device = device;
}

uint32_t SwapChain::getImageCount() const{
    return imageCount;
}

VkExtent2D SwapChain::getExtent() const{
    return extent;
}

VkFormat SwapChain::getFormat() const{
    return format;
}

VkSurfaceKHR SwapChain::getSurface() const{
    return surface;
}

GLFWwindow* SwapChain::getWindow(){
    return window;
}

std::vector<uint32_t> SwapChain::makeScreenshot(uint32_t i) const {
    std::vector<uint32_t> buffer(extent.height * extent.width, 0);

    Buffer stagingBuffer;
    buffer::create(device->instance, device->getLogical(), sizeof(uint32_t) * extent.width * extent.height, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer.instance, &stagingBuffer.memory);

    VkCommandBuffer commandBuffer = singleCommandBuffer::create(device->getLogical(),commandPool);
    texture::transitionLayout(commandBuffer, swapChainAttachments.instances[i].image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
    texture::copy(commandBuffer, swapChainAttachments.instances[i].image, stagingBuffer.instance, {extent.width, extent.height, 1}, 1);
    texture::transitionLayout(commandBuffer, swapChainAttachments.instances[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_REMAINING_MIP_LEVELS, 0, 1);
    singleCommandBuffer::submit(device->getLogical(),device->getQueue(0,0),commandPool,&commandBuffer);

    void* map = nullptr;
    VkResult result = vkMapMemory(device->getLogical(), stagingBuffer.memory, 0, sizeof(uint32_t) * buffer.size(), 0, &map);
    debug::checkResult(result, std::string("in file ") + std::string(__FILE__) + std::string(" in line ") + std::to_string(__LINE__));

    std::memcpy(buffer.data(), map, sizeof(uint32_t) * buffer.size());
    vkUnmapMemory(device->getLogical(), stagingBuffer.memory);

    stagingBuffer.destroy(device->getLogical());

    return buffer;
}

}
