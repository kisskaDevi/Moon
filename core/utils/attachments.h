#ifndef ATTACHMENTS_H
#define ATTACHMENTS_H

#include <vulkan.h>
#include <vector>

struct imageInfo{
    uint32_t                        Count;
    VkFormat                        Format;
    VkExtent2D                      Extent;
    VkSampleCountFlagBits           Samples;
};

struct attachment{
    VkImage image{VK_NULL_HANDLE};
    VkDeviceMemory imageMemory{VK_NULL_HANDLE};
    VkImageView imageView{VK_NULL_HANDLE};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
};

struct attachments{
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

void createAttachments(VkPhysicalDevice physicalDevice, VkDevice device, const imageInfo image, uint32_t attachmentsCount, attachments* pAttachments, VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |VK_IMAGE_USAGE_SAMPLED_BIT);

#endif // ATTACHMENTS_H
