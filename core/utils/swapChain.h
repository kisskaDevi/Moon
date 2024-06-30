#ifndef SWAPCHAIN_H
#define SWAPCHAIN_H

#include <vulkan.h>

#include "attachments.h"
#include "operations.h"
#include "device.h"

namespace moon::utils {

class SwapChain{
private:
    struct SwapChainAttachment {
        VkImage image{ VK_NULL_HANDLE };
        utils::vkDefault::ImageView imageView;
    };

    GLFWwindow*             window{ nullptr };
    const PhysicalDevice*   device{nullptr};
    VkSurfaceKHR            surface{ VK_NULL_HANDLE };
    ImageInfo               imageInfo;

    VkSwapchainKHR swapChainKHR{VK_NULL_HANDLE};
    std::vector<SwapChainAttachment> attachments;
    utils::vkDefault::CommandPool commandPool;

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
    const VkImageView& SwapChain::imageView(uint32_t i) const;

    uint32_t getImageCount() const;
    VkExtent2D getExtent() const;
    VkFormat getFormat() const;
    VkSurfaceKHR getSurface() const;
    GLFWwindow* getWindow();

    std::vector<uint32_t> makeScreenshot(uint32_t i = 0) const;
};

}
#endif // SWAPCHAIN_H
