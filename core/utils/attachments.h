#ifndef ATTACHMENTS_H
#define ATTACHMENTS_H

#include <vulkan.h>
#include <vector>

struct imageInfo{
    uint32_t                        Count;
    VkFormat                        Format;
    VkOffset2D                      Offset;
    VkExtent2D                      Extent;
    VkExtent2D                      frameBufferExtent;
    VkSampleCountFlagBits           Samples;
};

struct attachment
{
    VkImage image{VK_NULL_HANDLE};
    VkDeviceMemory imageMemory{VK_NULL_HANDLE};
    VkImageView imageView{VK_NULL_HANDLE};
    VkSampler sampler{VK_NULL_HANDLE};

    void deleteAttachment(VkDevice* device){
        if(image)       {vkDestroyImage(*device, image, nullptr); image = VK_NULL_HANDLE;}
        if(imageMemory) {vkFreeMemory(*device, imageMemory, nullptr); imageMemory = VK_NULL_HANDLE;}
        if(imageView)   {vkDestroyImageView(*device, imageView, nullptr);  imageView = VK_NULL_HANDLE;}
    }
    void deleteSampler(VkDevice *device){
        if(sampler) {vkDestroySampler(*device,sampler,nullptr); sampler = VK_NULL_HANDLE;}
    }
};

class attachments
{
public:
    std::vector<VkImage> image;
    std::vector<VkDeviceMemory> imageMemory;
    std::vector<VkImageView> imageView;
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
};

struct GBufferAttachments{
    attachments        position;
    attachments        normal;
    attachments        color;
    attachments        emission;

    GBufferAttachments();
    GBufferAttachments(const GBufferAttachments& other);
    GBufferAttachments& operator=(const GBufferAttachments& other);
};

struct DeferredAttachments{
    attachments         image;
    attachments         blur;
    attachments         bloom;
    attachments         depth;
    GBufferAttachments  GBuffer;

    DeferredAttachments();
    DeferredAttachments(const DeferredAttachments& other);
    DeferredAttachments& operator=(const DeferredAttachments& other);

    void deleteAttachment(VkDevice device);
    void deleteSampler(VkDevice device);

    static size_t getGBufferOffset() {return 4;}
};

#endif // ATTACHMENTS_H
