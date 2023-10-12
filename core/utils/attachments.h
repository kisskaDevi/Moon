#ifndef ATTACHMENTS_H
#define ATTACHMENTS_H

#include <vulkan.h>
#include <vector>
#include <string>

struct imageInfo{
    uint32_t                        Count;
    VkFormat                        Format;
    VkOffset2D                      Offset;
    VkExtent2D                      Extent;
    VkExtent2D                      frameBufferExtent;
    VkSampleCountFlagBits           Samples;
};

class attachments
{
public:
    struct attachment{
        VkImage image{VK_NULL_HANDLE};
        VkDeviceMemory imageMemory{VK_NULL_HANDLE};
        VkImageView imageView{VK_NULL_HANDLE};
    };

    std::vector<attachment> instances;
    VkSampler sampler{VK_NULL_HANDLE};
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkClearValue clearValue{};

    attachments() = default;
    attachments(const attachments& other);
    attachments& operator=(const attachments& other);

    ~attachments() = default;
    void deleteAttachment(VkDevice device);
    void deleteSampler(VkDevice device);

    VkResult create(VkPhysicalDevice physicalDevice, VkDevice device, VkFormat format, VkImageUsageFlags usage, VkExtent2D extent, uint32_t count);
    VkResult createDepth(VkPhysicalDevice physicalDevice, VkDevice device, VkFormat format, VkImageUsageFlags usage, VkExtent2D extent, uint32_t count);

    static VkAttachmentDescription imageDescription(VkFormat format);
    static VkAttachmentDescription imageDescription(VkFormat format, VkImageLayout layout);
    static VkAttachmentDescription depthStencilDescription(VkFormat format);
    static VkAttachmentDescription depthDescription(VkFormat format);

    std::vector<VkImage> getImages() const;
};

#endif // ATTACHMENTS_H
