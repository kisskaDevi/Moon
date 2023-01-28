#include "operations.h"
#include <libs/glfw-3.3.4.bin.WIN64/include/GLFW/glfw3.h>

#include <unordered_map>
#include <set>
#include <utility>
#include <fstream>
#include <algorithm>
#include <iostream>

void PhysicalDevice::printMemoryProperties(VkPhysicalDeviceMemoryProperties memoryProperties){
    std::cout << "memoryHeapCount = " << memoryProperties.memoryHeapCount << std::endl;
    for (uint32_t i = 0; i < memoryProperties.memoryHeapCount; i++){
        std::cout << "heapFlag[" << i << "] = " << memoryProperties.memoryHeaps[i].flags << "\t\t"
                  << "heapSize[" << i << "] = " << memoryProperties.memoryHeaps[i].size << std::endl;
    }
    std::cout << "memoryTypeCount = " << memoryProperties.memoryTypeCount << std::endl;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++){
        std::cout << "heapIndex[" << i << "] = " << memoryProperties.memoryTypes[i].heapIndex << '\t'
                  << "heapType [" << i << "] = " << memoryProperties.memoryTypes[i].propertyFlags << std::endl;
    }
    std::cout<<std::endl;
}

uint32_t PhysicalDevice::findMemoryTypeIndex(VkPhysicalDevice physicalDevice, uint32_t memoryTypeBits, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++){
        if ((memoryTypeBits & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties){
            memoryTypeIndex = i; break;
        }
    }
    return memoryTypeIndex;
}

std::vector<QueueFamilyIndices> PhysicalDevice::findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    std::vector<QueueFamilyIndices> indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (uint32_t index = 0; index < queueFamilyCount; index++){
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &presentSupport);

        if ((queueFamilies[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) && presentSupport){
            indices.push_back({index,index});
        }
    }

    return indices;
}

VkSampleCountFlagBits PhysicalDevice::queryingSampleCount(VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties physicalDeviceProperties{};
    vkGetPhysicalDeviceProperties(device, &physicalDeviceProperties);

    VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
    if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
    if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
    if (counts & VK_SAMPLE_COUNT_8_BIT)  { return VK_SAMPLE_COUNT_8_BIT; }
    if (counts & VK_SAMPLE_COUNT_4_BIT)  { return VK_SAMPLE_COUNT_4_BIT; }
    if (counts & VK_SAMPLE_COUNT_2_BIT)  { return VK_SAMPLE_COUNT_2_BIT; }

    return VK_SAMPLE_COUNT_1_BIT;
}

bool PhysicalDevice::isSuitable(VkPhysicalDevice device, VkSurfaceKHR surface, const std::vector<const char*>& deviceExtensions)
{
    VkPhysicalDeviceFeatures supportedFeatures{};
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

    return PhysicalDevice::isExtensionsSupport(device, deviceExtensions) && SwapChain::queryingSupport(device, surface).isNotEmpty() && supportedFeatures.samplerAnisotropy;
}

bool PhysicalDevice::isExtensionsSupport(VkPhysicalDevice device, const std::vector<const char*>& deviceExtensions)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

VkResult Buffer::create(VkPhysicalDevice physicalDevice, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer* buffer, VkDeviceMemory* bufferMemory)
{
    VkResult result = VK_SUCCESS;

    VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    result = vkCreateBuffer(device, &bufferInfo, nullptr, buffer);

    if (result != VK_SUCCESS){
        std::cout << "VkBuffer " << *buffer << ": vkCreateBuffer result = " << result << std::endl;
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, *buffer, &memoryRequirements);

    VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memoryRequirements.size;
        allocInfo.memoryTypeIndex = PhysicalDevice::findMemoryTypeIndex(physicalDevice, memoryRequirements.memoryTypeBits, properties);
    result = vkAllocateMemory(device, &allocInfo, nullptr, bufferMemory);

    if (result != VK_SUCCESS){
        std::cout << "VkDeviceMemory " << *bufferMemory << ": vkAllocateMemory result = " << result << std::endl;
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, *buffer, *bufferMemory, 0);

    return result;
}

void Buffer::copy(VkCommandBuffer commandBuffer, VkDeviceSize size, VkBuffer srcBuffer, VkBuffer dstBuffer)
{
    VkBufferCopy copyRegion{};
        copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
}

VkCommandBuffer SingleCommandBuffer::create(VkDevice device, VkCommandPool commandPool)
{
    VkCommandBuffer commandBuffer;

    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

VkResult SingleCommandBuffer::submit(VkDevice device, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer* commandBuffer)
{
    VkResult result = VK_SUCCESS;

    vkEndCommandBuffer(*commandBuffer);

    VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = commandBuffer;
    result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);

    if (result != VK_SUCCESS){
        std::cout << "VkQueue " << queue << ": vkQueueSubmit result = " << result << std::endl;
        throw std::runtime_error("failed to submit command buffer!");
    }

    result = vkQueueWaitIdle(queue);

    if (result != VK_SUCCESS){
        std::cout << "VkQueue " << queue << ": vkWaitForFences result = " << result << std::endl;
        throw std::runtime_error("failed to wait fence!");
    }

    vkFreeCommandBuffers(device, commandPool, 1, commandBuffer);

    return result;
}

void Texture::transitionLayout(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, uint32_t baseArrayLayer, uint32_t arrayLayers){
    std::unordered_map<VkImageLayout,std::pair<VkAccessFlags,VkPipelineStageFlags>> layoutDescription;
    layoutDescription[VK_IMAGE_LAYOUT_UNDEFINED] = {0,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    layoutDescription[VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL] = {VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT};
    layoutDescription[VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL] = {VK_ACCESS_TRANSFER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT};
    layoutDescription[VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL] = {VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};

    VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = baseArrayLayer;
        barrier.subresourceRange.layerCount = arrayLayers;
        barrier.srcAccessMask = layoutDescription[oldLayout].first;
        barrier.dstAccessMask = layoutDescription[newLayout].first;
    vkCmdPipelineBarrier(commandBuffer, layoutDescription[oldLayout].second, layoutDescription[newLayout].second, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void Texture::copyFromBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, VkExtent3D extent, uint32_t layerCount){
    VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = layerCount;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = extent;
    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

VkResult Texture::create(VkPhysicalDevice physicalDevice, VkDevice device, VkImageCreateFlags flags, VkExtent3D extent, uint32_t arrayLayers, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageLayout layout, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage* image, VkDeviceMemory* imageMemory){
    VkResult result = VK_SUCCESS;

    VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.flags = flags;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent = extent;
        imageInfo.mipLevels = mipLevels;
        imageInfo.arrayLayers = arrayLayers;
        imageInfo.format = format;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = layout;
        imageInfo.usage = usage;
        imageInfo.samples = numSamples;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    result = vkCreateImage(device, &imageInfo, nullptr, image);

    if (result != VK_SUCCESS){
        std::cout << "VkImage " << *image << ": vkCreateImage result = " << result << std::endl;
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, *image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = PhysicalDevice::findMemoryTypeIndex(physicalDevice, memRequirements.memoryTypeBits, properties);
    result = vkAllocateMemory(device, &allocInfo, nullptr, imageMemory);

    if (result != VK_SUCCESS){
        std::cout << "VkDeviceMemory " << *imageMemory << ": vkAllocateMemory result = " << result << std::endl;
        throw std::runtime_error("failed to allocate memory!");
    }

    result = vkBindImageMemory(device, *image, *imageMemory, 0);

    if (result != VK_SUCCESS){
        std::cout << "VkImage " << *image << ": vkBindImageMemory result = " << result << std::endl;
        throw std::runtime_error("failed to bind image memory!");
    }

    return result;
}

VkResult Texture::createView(VkDevice device, VkImageViewType type, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels, uint32_t baseArrayLayer, uint32_t layerCount, VkImage image, VkImageView* imageView)
{
    VkResult result = VK_SUCCESS;

    VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = type;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = baseArrayLayer;
        viewInfo.subresourceRange.layerCount = layerCount;
    result = vkCreateImageView(device, &viewInfo, nullptr, imageView);

    if (result != VK_SUCCESS){
        std::cout << "VkImageView " << *imageView << ": vkCreateImageView result = " << result << std::endl;
        throw std::runtime_error("failed to create image view!");
    }

    return result;
}

void Texture::generateMipmaps(VkPhysicalDevice physicalDevice, VkCommandBuffer commandBuffer, VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, uint32_t baseArrayLayer, uint32_t layerCount)
{
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)){
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = baseArrayLayer;
        barrier.subresourceRange.layerCount = layerCount;
        barrier.subresourceRange.levelCount = 1;

    for (uint32_t i = 1, mipWidth = texWidth, mipHeight = texHeight; i < mipLevels;
         i++, mipWidth /= (mipWidth > 1) ? 2 : 1, mipHeight /= (mipHeight > 1) ? 2 : 1) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        blitDown(commandBuffer,image,i - 1,image,i,mipWidth,mipHeight,baseArrayLayer,layerCount,2);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void Texture::blitDown(VkCommandBuffer commandBuffer, VkImage srcImage, uint32_t srcMipLevel, VkImage dstImage, uint32_t dstMipLevel, uint32_t width, uint32_t height, uint32_t baseArrayLayer, uint32_t layerCount, float blitFactor)
{
    VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {static_cast<int32_t>(width),static_cast<int32_t>(height),1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = srcMipLevel;
        blit.srcSubresource.baseArrayLayer = baseArrayLayer;
        blit.srcSubresource.layerCount = layerCount;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {static_cast<int32_t>(width/blitFactor),static_cast<int32_t>(height/blitFactor),1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = dstMipLevel;
        blit.dstSubresource.baseArrayLayer = baseArrayLayer;
        blit.dstSubresource.layerCount = layerCount;
    vkCmdBlitImage(commandBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
}

void Texture::blitUp(VkCommandBuffer commandBuffer, VkImage srcImage, uint32_t srcMipLevel, VkImage dstImage, uint32_t dstMipLevel, uint32_t width, uint32_t height, uint32_t baseArrayLayer, uint32_t layerCount, float blitFactor)
{
    VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {static_cast<int32_t>(width/blitFactor),static_cast<int32_t>(height/blitFactor),1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = srcMipLevel;
        blit.srcSubresource.baseArrayLayer = baseArrayLayer;
        blit.srcSubresource.layerCount = layerCount;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {static_cast<int32_t>(width),static_cast<int32_t>(height),1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = dstMipLevel;
        blit.dstSubresource.baseArrayLayer = baseArrayLayer;
        blit.dstSubresource.layerCount = layerCount;
    vkCmdBlitImage(commandBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
}

SwapChain::SupportDetails SwapChain::queryingSupport(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    SwapChain::SupportDetails details{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount = 0, presentModeCount = 0;

    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0){
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR SwapChain::queryingSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

VkPresentModeKHR SwapChain::queryingPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return availablePresentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D SwapChain::queryingExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities)
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D actualExtent = (capabilities.currentExtent.width != UINT32_MAX && capabilities.currentExtent.height != UINT32_MAX) ? capabilities.currentExtent :
    VkExtent2D{ actualExtent.width = std::clamp(static_cast<uint32_t>(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                actualExtent.height = std::clamp(static_cast<uint32_t>(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};

    return actualExtent;
}

VkFormat Image::supportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
    VkFormat supportedFormat = candidates[0];
    for (VkFormat format : candidates)
    {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if ((tiling == VK_IMAGE_TILING_OPTIMAL || tiling == VK_IMAGE_TILING_LINEAR) && (props.linearTilingFeatures & features) == features){
            supportedFormat = format; break;
        }
    }
    return supportedFormat;
}

VkFormat Image::depthStencilFormat(VkPhysicalDevice physicalDevice)
{
    return Image::supportedFormat(
        physicalDevice,
        {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

std::vector<char> ShaderModule::readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

VkShaderModule ShaderModule::create(VkDevice* device, const std::vector<char>& code)
{
    VkShaderModule shaderModule{VK_NULL_HANDLE};
    VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkResult result = vkCreateShaderModule(*device, &createInfo, nullptr, &shaderModule);
    if (result != VK_SUCCESS){
        std::cout << "VkShaderModule " << shaderModule << ": vkCreateShaderModule result = " << result << std::endl;
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}

//  decriptorSetLayout

void createObjectDescriptorSetLayout(VkDevice* device, VkDescriptorSetLayout* descriptorSetLayout)
{
    uint32_t index = 0;
    std::array<VkDescriptorSetLayoutBinding, 1> uniformBufferLayoutBinding{};
        uniformBufferLayoutBinding[index].binding = 0;
        uniformBufferLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniformBufferLayoutBinding[index].descriptorCount = 1;
        uniformBufferLayoutBinding[index].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uniformBufferLayoutBinding[index].pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo uniformBufferLayoutInfo{};
        uniformBufferLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        uniformBufferLayoutInfo.bindingCount = static_cast<uint32_t>(uniformBufferLayoutBinding.size());
        uniformBufferLayoutInfo.pBindings = uniformBufferLayoutBinding.data();
    if (vkCreateDescriptorSetLayout(*device, &uniformBufferLayoutInfo, nullptr, descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create base object uniform buffer descriptor set layout!");
}

void createSkyboxObjectDescriptorSetLayout(VkDevice* device, VkDescriptorSetLayout* descriptorSetLayout)
{
    std::vector<VkDescriptorSetLayoutBinding> uniformBufferLayoutBinding;
    uniformBufferLayoutBinding.push_back(VkDescriptorSetLayoutBinding{});
        uniformBufferLayoutBinding.back().binding = uniformBufferLayoutBinding.size() - 1;
        uniformBufferLayoutBinding.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniformBufferLayoutBinding.back().descriptorCount = 1;
        uniformBufferLayoutBinding.back().stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uniformBufferLayoutBinding.back().pImmutableSamplers = nullptr;
    uniformBufferLayoutBinding.push_back(VkDescriptorSetLayoutBinding{});
        uniformBufferLayoutBinding.back().binding = uniformBufferLayoutBinding.size() - 1;
        uniformBufferLayoutBinding.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        uniformBufferLayoutBinding.back().descriptorCount = 1;
        uniformBufferLayoutBinding.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        uniformBufferLayoutBinding.back().pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo uniformBufferLayoutInfo{};
        uniformBufferLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        uniformBufferLayoutInfo.bindingCount = static_cast<uint32_t>(uniformBufferLayoutBinding.size());
        uniformBufferLayoutInfo.pBindings = uniformBufferLayoutBinding.data();
    if (vkCreateDescriptorSetLayout(*device, &uniformBufferLayoutInfo, nullptr, descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create base object uniform buffer descriptor set layout!");
}

void createNodeDescriptorSetLayout(VkDevice* device, VkDescriptorSetLayout* descriptorSetLayout)
{
    uint32_t index = 0;
    std::array<VkDescriptorSetLayoutBinding, 1> uniformBlockLayoutBinding{};
        uniformBlockLayoutBinding[index].binding = 0;
        uniformBlockLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniformBlockLayoutBinding[index].descriptorCount = 1;
        uniformBlockLayoutBinding[index].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uniformBlockLayoutBinding[index].pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo uniformBlockLayoutInfo{};
        uniformBlockLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        uniformBlockLayoutInfo.bindingCount = static_cast<uint32_t>(uniformBlockLayoutBinding.size());
        uniformBlockLayoutInfo.pBindings = uniformBlockLayoutBinding.data();
    if (vkCreateDescriptorSetLayout(*device, &uniformBlockLayoutInfo, nullptr, descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create base uniform block descriptor set layout!");
}

void createMaterialDescriptorSetLayout(VkDevice* device, VkDescriptorSetLayout* descriptorSetLayout)
{
    uint32_t index = 0;
    std::array<VkDescriptorSetLayoutBinding, 5> materialLayoutBinding{};
    //baseColorTexture;
        materialLayoutBinding[index].binding = 0;
        materialLayoutBinding[index].descriptorCount = 1;
        materialLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        materialLayoutBinding[index].pImmutableSamplers = nullptr;
        materialLayoutBinding[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    index++;
    //metallicRoughnessTexture;
        materialLayoutBinding[index].binding = 1;
        materialLayoutBinding[index].descriptorCount = 1;
        materialLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        materialLayoutBinding[index].pImmutableSamplers = nullptr;
        materialLayoutBinding[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    index++;
    //normalTexture;
        materialLayoutBinding[index].binding = 2;
        materialLayoutBinding[index].descriptorCount = 1;
        materialLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        materialLayoutBinding[index].pImmutableSamplers = nullptr;
        materialLayoutBinding[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    index++;
    //occlusionTexture;
        materialLayoutBinding[index].binding = 3;
        materialLayoutBinding[index].descriptorCount = 1;
        materialLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        materialLayoutBinding[index].pImmutableSamplers = nullptr;
        materialLayoutBinding[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    index++;
    //emissiveTexture;
        materialLayoutBinding[index].binding = 4;
        materialLayoutBinding[index].descriptorCount = 1;
        materialLayoutBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        materialLayoutBinding[index].pImmutableSamplers = nullptr;
        materialLayoutBinding[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo materialLayoutInfo{};
        materialLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        materialLayoutInfo.bindingCount = static_cast<uint32_t>(materialLayoutBinding.size());
        materialLayoutInfo.pBindings = materialLayoutBinding.data();
    if (vkCreateDescriptorSetLayout(*device, &materialLayoutInfo, nullptr, descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create base material descriptor set layout!");
}

void createSpotLightDescriptorSetLayout(VkDevice* device, VkDescriptorSetLayout* descriptorSetLayout)
{
    uint32_t index = 0;
    std::array<VkDescriptorSetLayoutBinding,3> lihgtBinding{};
        lihgtBinding[index].binding = index;
        lihgtBinding[index].descriptorCount = 1;
        lihgtBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        lihgtBinding[index].stageFlags = VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT;
        lihgtBinding[index].pImmutableSamplers = nullptr;
    index++;
        lihgtBinding[index].binding = index;
        lihgtBinding[index].descriptorCount = 1;
        lihgtBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        lihgtBinding[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        lihgtBinding[index].pImmutableSamplers = nullptr;
    index++;
        lihgtBinding[index].binding = index;
        lihgtBinding[index].descriptorCount = 1;
        lihgtBinding[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        lihgtBinding[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        lihgtBinding[index].pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo lihgtLayoutInfo{};
        lihgtLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lihgtLayoutInfo.bindingCount = static_cast<uint32_t>(lihgtBinding.size());
        lihgtLayoutInfo.pBindings = lihgtBinding.data();
    if (vkCreateDescriptorSetLayout(*device, &lihgtLayoutInfo, nullptr, descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create SpotLightingPass descriptor set layout!");
}
