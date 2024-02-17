#ifndef SWAPCHAIN_H
#define SWAPCHAIN_H

#include <vulkan.h>

#include "attachments.h"
#include "operations.h"
#include "device.h"

class swapChain{
private:
    physicalDevice      device;
    uint32_t            imageCount{0};
    VkExtent2D          extent{0,0};
    VkFormat            format{VK_FORMAT_UNDEFINED};

    VkSwapchainKHR      swapChainKHR{VK_NULL_HANDLE};
    attachments         swapChainAttachments;

    GLFWwindow*         window{nullptr};
    VkSurfaceKHR        surface{VK_NULL_HANDLE};

    VkCommandPool       commandPool{VK_NULL_HANDLE};

public:
    swapChain() = default;
    void destroy();

    VkResult create(GLFWwindow* window, VkSurfaceKHR surface, uint32_t queueFamilyIndexCount, uint32_t* pQueueFamilyIndices, int32_t maxImageCount = -1);    
    void setDevice(const physicalDevice& device);

    VkSwapchainKHR& operator()();
    ::attachment& attachment(uint32_t i);

    uint32_t getImageCount() const;
    VkExtent2D getExtent() const;
    VkFormat getFormat() const;
    VkSurfaceKHR getSurface() const;
    GLFWwindow* getWindow();

    std::vector<uint32_t> makeScreenshot(uint32_t i = 0) const;
};

#endif // SWAPCHAIN_H
