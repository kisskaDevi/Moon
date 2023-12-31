#include "swapChain.h"
#include "buffer.h"
#include <glfw3.h>

void swapChain::destroy(){
    for (auto& instance: swapChainAttachments.instances){
        if(instance.imageView){
            vkDestroyImageView(device.getLogical(), instance.imageView, nullptr);
            instance.imageView = VK_NULL_HANDLE;
        }
    }

    if(swapChainKHR) {
        vkDestroySwapchainKHR(device.getLogical(), swapChainKHR, nullptr);
        swapChainKHR = VK_NULL_HANDLE;
    }

    if(commandPool) {
        vkDestroyCommandPool(device.getLogical(), commandPool, nullptr); commandPool = VK_NULL_HANDLE;
    }
}

VkResult swapChain::create(GLFWwindow* window, VkSurfaceKHR surface, uint32_t queueFamilyIndexCount, uint32_t* pQueueFamilyIndices, int32_t maxImageCount){
    VkResult result = VK_SUCCESS;

    this->window = window;
    this->surface = surface;

    SwapChain::SupportDetails swapChainSupport = SwapChain::queryingSupport(device.instance, surface);
    VkSurfaceFormatKHR surfaceFormat = SwapChain::queryingSurfaceFormat(swapChainSupport.formats);
    VkSurfaceCapabilitiesKHR capabilities = SwapChain::queryingSupport(device.instance, surface).capabilities;

    imageCount = SwapChain::queryingSupportImageCount(device.instance, surface);
    imageCount = (maxImageCount > 0 && imageCount > static_cast<uint32_t>(maxImageCount)) ? static_cast<uint32_t>(maxImageCount) : imageCount;
    extent = SwapChain::queryingExtent(window, capabilities);
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
        createInfo.presentMode = SwapChain::queryingPresentMode(swapChainSupport.presentModes);
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
    result = vkCreateSwapchainKHR(device.getLogical(), &createInfo, nullptr, &swapChainKHR);
    debug::checkResult(result, "VkSwapchainKHR : vkCreateSwapchainKHR result = " + std::to_string(result));

    swapChainAttachments.instances.resize(imageCount);
    std::vector<VkImage> images = swapChainAttachments.getImages();
    result = vkGetSwapchainImagesKHR(device.getLogical(), swapChainKHR, &imageCount, images.data());
    debug::checkResult(result, "VkSwapchainKHR : vkGetSwapchainImagesKHR result = " + std::to_string(result));

    for (auto& instance: swapChainAttachments.instances){
        instance.image = images[&instance - &swapChainAttachments.instances[0]];
        result = Texture::createView(   device.getLogical(),
                                        VK_IMAGE_VIEW_TYPE_2D,
                                        surfaceFormat.format,
                                        VK_IMAGE_ASPECT_COLOR_BIT,
                                        1,
                                        0,
                                        1,
                                        instance.image,
                                        &instance.imageView);
        debug::checkResult(result, "attachments::image : Texture::createView result = " + std::to_string(result));
    }

    VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = 0;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    result = vkCreateCommandPool(device.getLogical(), &poolInfo, nullptr, &commandPool);
    debug::checkResult(result, "VkCommandPool : vkCreateCommandPool result = " + std::to_string(result));

    return result;
}

VkSwapchainKHR& swapChain::operator()(){
    return swapChainKHR;
}

::attachment& swapChain::attachment(uint32_t i){
    return swapChainAttachments.instances[i];
}

void swapChain::setDevice(const physicalDevice& device){
    this->device = device;
}

uint32_t swapChain::getImageCount(){
    return imageCount;
}

VkExtent2D swapChain::getExtent(){
    return extent;
}

VkFormat swapChain::getFormat(){
    return format;
}

GLFWwindow* swapChain::getWindow(){
    return window;
}

VkSurfaceKHR swapChain::getSurface(){
    return surface;
}

std::vector<uint32_t> swapChain::makeScreenShot(uint32_t i) const {
    std::vector<uint32_t> buffer(extent.height * extent.height, 0);

    ::buffer stagingBuffer;
    Buffer::create(device.instance, device.getLogical(), sizeof(uint32_t) * extent.width * extent.height, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer.instance, &stagingBuffer.memory);

    VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
    Texture::transitionLayout(commandBuffer, swapChainAttachments.instances[i].image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
    Texture::copy(commandBuffer, swapChainAttachments.instances[i].image, stagingBuffer.instance, {extent.width, extent.height, 1}, 1);
    Texture::transitionLayout(commandBuffer, swapChainAttachments.instances[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_REMAINING_MIP_LEVELS, 0, 1);
    SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool,&commandBuffer);

    void* map = nullptr;
    vkMapMemory(device.getLogical(), stagingBuffer.memory, 0, sizeof(uint32_t) * buffer.size(), 0, &map);
    std::memcpy(buffer.data(), map, sizeof(uint32_t) * buffer.size());
    vkUnmapMemory(device.getLogical(), stagingBuffer.memory);

    stagingBuffer.destroy(device.getLogical());

    return buffer;
}

