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
    VkImage image;
    VkDeviceMemory imageMemory;
    VkImageView imageView;
    VkSampler sampler;

    void deleteAttachment(VkDevice* device)
    {
        vkDestroyImage(*device, image, nullptr);
        vkFreeMemory(*device, imageMemory, nullptr);
        vkDestroyImageView(*device, imageView, nullptr);
    }
    void deleteSampler(VkDevice *device)
    {
        vkDestroySampler(*device,sampler,nullptr);
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
    VkSampler sampler;

    attachments();
    ~attachments();

    attachments& operator=(const attachments& other);

    void resize(size_t size);
    void deleteAttachment(VkDevice * device);
    void deleteSampler(VkDevice * device);
    void setSize(size_t size);
    size_t getSize();
};

struct GBufferAttachments{
    attachments*        position;
    attachments*        normal;
    attachments*        color;
    attachments*        emission;
};

struct DeferredAttachments{
    attachments*        image;
    attachments*        blur;
    attachments*        bloom;
    attachment*         depth;
    GBufferAttachments  GBuffer;
};

#endif // ATTACHMENTS_H
