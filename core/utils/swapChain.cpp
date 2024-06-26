#include "swapChain.h"
#include "buffer.h"
#include <glfw3.h>
#include <cstring>

namespace moon::utils {

SwapChain::SwapChain(SwapChain&& other) {
    swap(other);
}

SwapChain& SwapChain::operator=(SwapChain&& other) {
    swap(other);
    return *this;
}

void SwapChain::swap(SwapChain& other) {
    std::swap(window, other.window);
    std::swap(device, other.device);
    std::swap(surface, other.surface);
    std::swap(imageInfo, other.imageInfo);
    std::swap(swapChainKHR, other.swapChainKHR);
    std::swap(commandPool, other.commandPool);
    std::swap(attachments, other.attachments);
}

VkResult SwapChain::create(const PhysicalDevice* device, GLFWwindow* window, VkSurfaceKHR surface, std::vector<uint32_t> queueFamilyIndices, int32_t maxImageCount){
    this->device = device;
    this->window = window;
    this->surface = surface;

    swapChain::SupportDetails supportDetails = swapChain::queryingSupport(device->instance, surface);
    VkSurfaceFormatKHR surfaceFormat = swapChain::queryingSurfaceFormat(supportDetails.formats);
    VkSurfaceCapabilitiesKHR capabilities = swapChain::queryingSupport(device->instance, surface).capabilities;

    imageInfo.Count = swapChain::queryingSupportImageCount(device->instance, surface);
    imageInfo.Count = (maxImageCount > 0 && imageInfo.Count > static_cast<uint32_t>(maxImageCount)) ? static_cast<uint32_t>(maxImageCount) : imageInfo.Count;
    imageInfo.Extent = swapChain::queryingExtent(window, capabilities);
    imageInfo.Format = surfaceFormat.format;

    VkResult result = VK_SUCCESS;
    CHECK(result = swapChainKHR.create(device->getLogical(), imageInfo, supportDetails, queueFamilyIndices, surface, surfaceFormat));

    for (const auto& image: swapChainKHR.images()){
        auto& attachment = attachments.emplace_back();
        attachment.image = image;
        CHECK(result = attachment.imageView.create(device->getLogical(), attachment.image, VK_IMAGE_VIEW_TYPE_2D, surfaceFormat.format, VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1));
    }

    CHECK(result = commandPool.create(device->getLogical()));

    return result;
}

VkResult SwapChain::present(VkSemaphore waitSemaphore, uint32_t imageIndex) const {
    VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &waitSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChainKHR;
        presentInfo.pImageIndices = &imageIndex;
    return vkQueuePresentKHR(device->getQueue(0, 0), &presentInfo);
}

SwapChain::operator const VkSwapchainKHR&() const {
    return swapChainKHR;
}

const VkImageView& SwapChain::imageView(uint32_t i) const {
    return attachments[i].imageView;
}

ImageInfo SwapChain::info() const { return imageInfo;}
VkSurfaceKHR SwapChain::getSurface() const { return surface;}
GLFWwindow* SwapChain::getWindow() const { return window;}

std::vector<uint32_t> SwapChain::makeScreenshot(uint32_t i) const {
    std::vector<uint32_t> buffer(imageInfo.Extent.height * imageInfo.Extent.width, 0);

    Buffer cache;
    cache.create(device->instance, device->getLogical(), sizeof(uint32_t) * imageInfo.Extent.width * imageInfo.Extent.height, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkCommandBuffer commandBuffer = singleCommandBuffer::create(device->getLogical(),commandPool);
    texture::transitionLayout(commandBuffer, attachments[i].image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
    texture::copy(commandBuffer, attachments[i].image, cache, { imageInfo.Extent.width, imageInfo.Extent.height, 1}, 1);
    texture::transitionLayout(commandBuffer, attachments[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_REMAINING_MIP_LEVELS, 0, 1);
    singleCommandBuffer::submit(device->getLogical(),device->getQueue(0,0),commandPool,&commandBuffer);

    void* map = nullptr;
    VkResult result = vkMapMemory(device->getLogical(), cache, 0, sizeof(uint32_t) * buffer.size(), 0, &map);
    debug::checkResult(result, std::string("in file ") + std::string(__FILE__) + std::string(" in line ") + std::to_string(__LINE__));

    std::memcpy(buffer.data(), map, sizeof(uint32_t) * buffer.size());
    vkUnmapMemory(device->getLogical(), cache);

    return buffer;
}

}
