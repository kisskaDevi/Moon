#include "attachments.h"
#include "operations.h"
#include <algorithm>
#include <iterator>

attachments::attachments(const attachments &other)
{
    std::copy(other.image.begin(), other.image.end(), std::back_inserter(image));
    std::copy(other.imageMemory.begin(), other.imageMemory.end(), std::back_inserter(imageMemory));
    std::copy(other.imageView.begin(), other.imageView.end(), std::back_inserter(imageView));
    sampler = other.sampler;
    format = other.format;
}

attachments& attachments::operator=(const attachments& other)
{
    std::copy(other.image.begin(), other.image.end(), std::back_inserter(image));
    std::copy(other.imageMemory.begin(), other.imageMemory.end(), std::back_inserter(imageMemory));
    std::copy(other.imageView.begin(), other.imageView.end(), std::back_inserter(imageView));
    sampler = other.sampler;
    format = other.format;

    return *this;
}

void attachments::create(VkPhysicalDevice physicalDevice, VkDevice device, VkFormat format, VkImageUsageFlags usage, VkExtent2D extent, uint32_t count)
{
    image.resize(count);
    imageMemory.resize(count);
    imageView.resize(count);

    this->format = format;
    clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

    auto instance = image.begin();
    auto memory = imageMemory.begin();
    auto view = imageView.begin();
    for(; instance != image.end() || memory != imageMemory.end() || view != imageView.end(); instance++, memory++, view++){
        Texture::create(    physicalDevice,
                            device,
                            0,
                            {extent.width,extent.height,1},
                            1,
                            1,
                            VK_SAMPLE_COUNT_1_BIT,
                            format,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            usage,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            &(*instance),
                            &(*memory));

        Texture::createView(    device,
                                VK_IMAGE_VIEW_TYPE_2D,
                                format,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                1,
                                0,
                                1,
                                *instance,
                                &(*view));
    }
}

void attachments::createDepth(VkPhysicalDevice physicalDevice, VkDevice device, VkFormat format, VkImageUsageFlags usage, VkExtent2D extent, uint32_t count)
{
    image.resize(count);
    imageMemory.resize(count);
    imageView.resize(count);

    this->format = format;
    clearValue.depthStencil = {1.0f, 0};

    auto instance = image.begin();
    auto memory = imageMemory.begin();
    auto view = imageView.begin();
    for(; instance != image.end() || memory != imageMemory.end() || view != imageView.end(); instance++, memory++, view++){
        Texture::create(    physicalDevice,
                            device,
                            0,
                            {extent.width,extent.height,1},
                            1,
                            1,
                            VK_SAMPLE_COUNT_1_BIT,
                            format,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            usage,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            &(*instance),
                            &(*memory));

        Texture::createView(    device,
                                VK_IMAGE_VIEW_TYPE_2D,
                                format,
                                VK_IMAGE_ASPECT_DEPTH_BIT,
                                1,
                                0,
                                1,
                                *instance,
                                &(*view));
    }
}

void attachments::deleteAttachment(VkDevice device)
{
    std::for_each(image.begin(), image.end(), [&device](VkImage& image){ vkDestroyImage(device, image, nullptr); image = VK_NULL_HANDLE;});
    std::for_each(imageMemory.begin(), imageMemory.end(), [&device](VkDeviceMemory& memory){ vkFreeMemory(device, memory, nullptr); memory = VK_NULL_HANDLE;});
    std::for_each(imageView.begin(), imageView.end(), [&device](VkImageView& view){ vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE;});
    image.clear();
    imageMemory.clear();
    imageView.clear();
}

void attachments::deleteSampler(VkDevice device)
{
    if(sampler){ vkDestroySampler(device,sampler,nullptr); sampler = VK_NULL_HANDLE;}
}

VkAttachmentDescription attachments::imageDescription(VkFormat format)
{
    VkAttachmentDescription description{};
    description.format = format;
    description.samples = VK_SAMPLE_COUNT_1_BIT;
    description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    description.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return description;
}

VkAttachmentDescription attachments::imageDescription(VkFormat format, VkImageLayout layout)
{
    VkAttachmentDescription description{};
    description.format = format;
    description.samples = VK_SAMPLE_COUNT_1_BIT;
    description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    description.finalLayout = layout;
    return description;
}

VkAttachmentDescription attachments::depthDescription(VkFormat format)
{
    VkAttachmentDescription description{};
    description.format = format;
    description.samples = VK_SAMPLE_COUNT_1_BIT;
    description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    description.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    description.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    description.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return description;
}

VkAttachmentDescription attachments::depthStencilDescription(VkFormat format)
{
    VkAttachmentDescription description{};
    description.format = format;
    description.samples = VK_SAMPLE_COUNT_1_BIT;
    description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    description.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    description.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return description;
}

GBufferAttachments::GBufferAttachments()
{}

GBufferAttachments::GBufferAttachments(const GBufferAttachments& other):
      position(other.position),
      normal(other.normal),
      color(other.color),
      emission(other.emission)
{}

GBufferAttachments& GBufferAttachments::operator=(const GBufferAttachments& other)
{
    position = other.position;
    normal = other.normal;
    color = other.color;
    emission = other.emission;
    return *this;
}

DeferredAttachments::DeferredAttachments()
{}

DeferredAttachments::DeferredAttachments(const DeferredAttachments& other):
    image(other.image),
    blur(other.blur),
    bloom(other.bloom),
    depth(other.depth),
    GBuffer(other.GBuffer)
{}

DeferredAttachments& DeferredAttachments::operator=(const DeferredAttachments& other)
{
    image = other.image;
    blur = other.blur;
    bloom = other.bloom;
    depth = other.depth;
    GBuffer = other.GBuffer;
    return *this;
}

void DeferredAttachments::deleteAttachment(VkDevice device){
    image.deleteAttachment(device);
    blur.deleteAttachment(device);
    bloom.deleteAttachment(device);
    depth.deleteAttachment(device);
    GBuffer.color.deleteAttachment(device);
    GBuffer.emission.deleteAttachment(device);
    GBuffer.normal.deleteAttachment(device);
    GBuffer.position.deleteAttachment(device);
}

void DeferredAttachments::deleteSampler(VkDevice device){
    image.deleteSampler(device);
    blur.deleteSampler(device);
    bloom.deleteSampler(device);
    depth.deleteSampler(device);
    GBuffer.color.deleteSampler(device);
    GBuffer.emission.deleteSampler(device);
    GBuffer.normal.deleteSampler(device);
    GBuffer.position.deleteSampler(device);
}
