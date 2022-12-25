#ifndef ATTACHMENTS_H
#define ATTACHMENTS_H

#include <libs/vulkan/vulkan.h>
#include <vector>

struct imageInfo{
    uint32_t                        Count;
    VkFormat                        Format;
    VkExtent2D                      Extent;
    VkSampleCountFlagBits           Samples;
};

struct attachment
{
    VkImage image{VK_NULL_HANDLE};
    VkDeviceMemory imageMemory{VK_NULL_HANDLE};
    VkImageView imageView{VK_NULL_HANDLE};
    VkSampler sampler{VK_NULL_HANDLE};

    void deleteAttachment(VkDevice* device)
    {
        if(image)       vkDestroyImage(*device, image, nullptr);
        if(imageMemory) vkFreeMemory(*device, imageMemory, nullptr);
        if(imageView)   vkDestroyImageView(*device, imageView, nullptr);
    }
    void deleteSampler(VkDevice *device)
    {
        if(sampler) vkDestroySampler(*device,sampler,nullptr);
    }
};

class attachments
{
private:
    size_t size;
public:
    std::vector<VkImage> image;
    std::vector<VkDeviceMemory> imageMemory;
    std::vector<VkImageView> imageView;
    VkSampler sampler{VK_NULL_HANDLE};

    attachments();
    ~attachments();

    attachments(const attachments& other);
    attachments& operator=(const attachments& other);

    void resize(size_t size);
    void deleteAttachment(VkDevice * device);
    void deleteSampler(VkDevice * device);
    size_t getSize();
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
    attachment          depth;
    GBufferAttachments  GBuffer;

    DeferredAttachments();
    DeferredAttachments(const DeferredAttachments& other);
    DeferredAttachments& operator=(const DeferredAttachments& other);

    void deleteAttachment(VkDevice * device);
    void deleteSampler(VkDevice * device);
};

#endif // ATTACHMENTS_H
