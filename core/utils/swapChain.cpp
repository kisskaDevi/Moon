#include "swapChain.h"

#include <glfw3.h>

#ifndef NDEBUG
#include <iostream>
#endif

swapChain::swapChain()
{}

void swapChain::destroy(){
    for(size_t i=0; i < imageCount; i++){
        if(swapChainAttachments.imageView[i]){
            vkDestroyImageView(device, swapChainAttachments.imageView[i], nullptr);
        }
    }

    if(swapChainKHR) {vkDestroySwapchainKHR(device, swapChainKHR, nullptr); swapChainKHR = VK_NULL_HANDLE;}
}

VkResult swapChain::create(GLFWwindow* window, VkSurfaceKHR surface, uint32_t queueFamilyIndexCount, uint32_t* pQueueFamilyIndices, int32_t maxImageCount){
    VkResult result = VK_SUCCESS;

    this->window = window;
    this->surface = surface;

    SwapChain::SupportDetails swapChainSupport = SwapChain::queryingSupport(physicalDevice, surface);
    VkSurfaceFormatKHR surfaceFormat = SwapChain::queryingSurfaceFormat(swapChainSupport.formats);
    VkSurfaceCapabilitiesKHR capabilities = SwapChain::queryingSupport(physicalDevice, surface).capabilities;

    imageCount = SwapChain::queryingSupportImageCount(physicalDevice, surface);
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
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT ;
        createInfo.imageSharingMode = queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
        createInfo.pQueueFamilyIndices = pQueueFamilyIndices;
        createInfo.queueFamilyIndexCount = queueFamilyIndexCount;
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = SwapChain::queryingPresentMode(swapChainSupport.presentModes);
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
    result = vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChainKHR);
    debug::checkResult(result, "VkSwapchainKHR : vkCreateSwapchainKHR result = " + std::to_string(result));

    swapChainAttachments.image.resize(imageCount);
    swapChainAttachments.imageView.resize(imageCount);
    result = vkGetSwapchainImagesKHR(device, swapChainKHR, &imageCount, swapChainAttachments.image.data());
    debug::checkResult(result, "VkSwapchainKHR : vkGetSwapchainImagesKHR result = " + std::to_string(result));

    for (size_t size = 0; size < swapChainAttachments.imageView.size(); size++){
        result = Texture::createView(   device,
                                        VK_IMAGE_VIEW_TYPE_2D,
                                        surfaceFormat.format,
                                        VK_IMAGE_ASPECT_COLOR_BIT,
                                        1,
                                        0,
                                        1,
                                        swapChainAttachments.image[size],
                                        &swapChainAttachments.imageView[size]);
        debug::checkResult(result, "attachments::image : Texture::createView result = " + std::to_string(result));
    }
    return result;
}

VkSwapchainKHR& swapChain::operator()(){
    return swapChainKHR;
}

attachments& swapChain::attachment(){
    return swapChainAttachments;
}

void swapChain::setDevice(VkPhysicalDevice physicalDevice, VkDevice device){
    this->physicalDevice = physicalDevice;
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
