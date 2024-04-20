#ifndef ATTACHMENTS_H
#define ATTACHMENTS_H

#include <vulkan.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <optional>

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

class texture;

struct attachmentsDatabase{
    struct data{
        bool enable{false};
        attachments* pImages{nullptr};
    };

    std::string defaultEmptyTexture;
    std::unordered_map<std::string, texture*> emptyTexturesMap;
    std::unordered_map<std::string, data> attachmentsMap;

    attachmentsDatabase() = default;
    attachmentsDatabase(const std::string& emptyTextureId, texture* emptyTexture);
    attachmentsDatabase(const attachmentsDatabase&) = default;
    attachmentsDatabase& operator=(const attachmentsDatabase&) = default;

    void destroy();

    bool addEmptyTexture(const std::string& id, texture* emptyTexture);
    bool addAttachmentData(const std::string& id, bool enable, attachments* pImages);
    bool enable(const std::string& id) const;
    const attachments* get(const std::string& id) const;
    VkImageView imageView(const std::string& id, const uint32_t imageIndex, const std::optional<std::string>& emptyTextureId = std::nullopt) const;
    VkSampler sampler(const std::string& id, const std::optional<std::string>& emptyTextureId = std::nullopt) const;
    VkDescriptorImageInfo descriptorImageInfo(const std::string& id, const uint32_t imageIndex, const std::optional<std::string>& emptyTextureId = std::nullopt) const;
};

#endif // ATTACHMENTS_H
