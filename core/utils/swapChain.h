#ifndef SWAPCHAIN_H
#define SWAPCHAIN_H

#include <vulkan.h>

#include "attachments.h"
#include "operations.h"
#include "device.h"

namespace moon::utils {

class SwapChain{
private:
    const PhysicalDevice* device{nullptr};
    uint32_t            imageCount{0};
    VkExtent2D          extent{0,0};
    VkFormat            format{VK_FORMAT_UNDEFINED};

    VkSwapchainKHR      swapChainKHR{VK_NULL_HANDLE};
    Attachments         swapChainAttachments;

    GLFWwindow*         window{nullptr};
    VkSurfaceKHR        surface{VK_NULL_HANDLE};

    VkCommandPool       commandPool{VK_NULL_HANDLE};

    void destroy();
public:
    SwapChain() = default;
    ~SwapChain();
    SwapChain(const SwapChain&) = delete;
    SwapChain& operator=(const SwapChain&) = delete;
    SwapChain(SwapChain&&) = default;
    SwapChain& operator=(SwapChain&&) = default;

    VkResult create(const PhysicalDevice* device, GLFWwindow* window, VkSurfaceKHR surface, std::vector<uint32_t> queueFamilyIndices, int32_t maxImageCount = -1);
    VkResult present(VkSemaphore waitSemaphore, uint32_t imageIndex) const;

    operator VkSwapchainKHR&();
    Attachment& attachment(uint32_t i);

    uint32_t getImageCount() const;
    VkExtent2D getExtent() const;
    VkFormat getFormat() const;
    VkSurfaceKHR getSurface() const;
    GLFWwindow* getWindow();

    std::vector<uint32_t> makeScreenshot(uint32_t i = 0) const;
};

}
#endif // SWAPCHAIN_H
