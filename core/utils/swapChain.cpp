#include "swapChain.h"
#include "buffer.h"
#include <glfw3.h>
#include <cstring>

namespace moon::utils {

void SwapChain::destroy(){
    if(swapChainKHR) {
        vkDestroySwapchainKHR(device->getLogical(), swapChainKHR, nullptr);
        swapChainKHR = VK_NULL_HANDLE;
    }

    if(commandPool) {
        vkDestroyCommandPool(device->getLogical(), commandPool, nullptr); commandPool = VK_NULL_HANDLE;
    }

    attachments.clear();
}

SwapChain::~SwapChain() {
    destroy();
}

VkResult SwapChain::create(const PhysicalDevice* device, GLFWwindow* window, VkSurfaceKHR surface, std::vector<uint32_t> queueFamilyIndices, int32_t maxImageCount){
    destroy();
    this->device = device;

    VkResult result = VK_SUCCESS;

    this->window = window;
    this->surface = surface;

    swapChain::SupportDetails swapChainSupport = swapChain::queryingSupport(device->instance, surface);
    VkSurfaceFormatKHR surfaceFormat = swapChain::queryingSurfaceFormat(swapChainSupport.formats);
    VkSurfaceCapabilitiesKHR capabilities = swapChain::queryingSupport(device->instance, surface).capabilities;

    imageInfo.Count = swapChain::queryingSupportImageCount(device->instance, surface);
    imageInfo.Count = (maxImageCount > 0 && imageInfo.Count > static_cast<uint32_t>(maxImageCount)) ? static_cast<uint32_t>(maxImageCount) : imageInfo.Count;
    imageInfo.Extent = swapChain::queryingExtent(window, capabilities);
    imageInfo.Format = surfaceFormat.format;

    VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageInfo.Count;
        createInfo.imageFormat = imageInfo.Format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = imageInfo.Extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        createInfo.imageSharingMode = queueFamilyIndices.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
        createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
        createInfo.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size());
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = swapChain::queryingPresentMode(swapChainSupport.presentModes);
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
    CHECK(result = vkCreateSwapchainKHR(device->getLogical(), &createInfo, nullptr, &swapChainKHR));

    attachments.resize(imageInfo.Count);
    std::vector<VkImage> images;
    for (auto& attachment : attachments) {
        images.push_back(attachment.image);
    }
    CHECK(result = vkGetSwapchainImagesKHR(device->getLogical(), swapChainKHR, &imageInfo.Count, images.data()));

    for (auto& attachment: attachments){
        attachment.device = device->getLogical();
        attachment.image = images[&attachment - &attachments[0]];
        result = texture::createView(   device->getLogical(),
                                        VK_IMAGE_VIEW_TYPE_2D,
                                        surfaceFormat.format,
                                        VK_IMAGE_ASPECT_COLOR_BIT,
                                        1,
                                        0,
                                        1,
                                        attachment.image,
                                        &attachment.imageView);
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

VkResult SwapChain::present(VkSemaphore waitSemaphore, uint32_t imageIndex) const {
    VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &waitSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapChainKHR;
        presentInfo.pImageIndices = &imageIndex;
    return vkQueuePresentKHR(device->getQueue(0, 0), &presentInfo);

}

SwapChain::operator VkSwapchainKHR&(){
    return swapChainKHR;
}

const VkImageView& SwapChain::imageView(uint32_t i) const {
    return attachments[i].imageView;
}

uint32_t SwapChain::getImageCount() const{
    return imageInfo.Count;
}

VkExtent2D SwapChain::getExtent() const{
    return imageInfo.Extent;
}

VkFormat SwapChain::getFormat() const{
    return imageInfo.Format;
}

VkSurfaceKHR SwapChain::getSurface() const{
    return surface;
}

GLFWwindow* SwapChain::getWindow(){
    return window;
}

std::vector<uint32_t> SwapChain::makeScreenshot(uint32_t i) const {
    std::vector<uint32_t> buffer(imageInfo.Extent.height * imageInfo.Extent.width, 0);

    Buffer stagingBuffer;
    buffer::create(device->instance, device->getLogical(), sizeof(uint32_t) * imageInfo.Extent.width * imageInfo.Extent.height, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer.instance, &stagingBuffer.memory);

    VkCommandBuffer commandBuffer = singleCommandBuffer::create(device->getLogical(),commandPool);
    texture::transitionLayout(commandBuffer, attachments[i].image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
    texture::copy(commandBuffer, attachments[i].image, stagingBuffer.instance, { imageInfo.Extent.width, imageInfo.Extent.height, 1}, 1);
    texture::transitionLayout(commandBuffer, attachments[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_REMAINING_MIP_LEVELS, 0, 1);
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
