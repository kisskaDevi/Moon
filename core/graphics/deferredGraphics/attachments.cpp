#include "attachments.h"
#include "core/operations.h"

attachments::attachments()
{}

attachments::~attachments()
{}

attachments::attachments(const attachments &other)
{
    image.resize(other.size);
    imageMemory.resize(other.size);
    imageView.resize(other.size);
    for(size_t i=0;i<other.size;i++){
        image[i] = other.image[i];
        imageMemory[i] = other.imageMemory[i];
        imageView[i] = other.imageView[i];
    }
    sampler = other.sampler;
    size = other.size;

    format = other.format;
}

attachments& attachments::operator=(const attachments &other)
{
    image.resize(other.size);
    imageMemory.resize(other.size);
    imageView.resize(other.size);
    for(size_t i=0;i<other.size;i++){
        image[i] = other.image[i];
        imageMemory[i] = other.imageMemory[i];
        imageView[i] = other.imageView[i];
    }
    sampler = other.sampler;
    size = other.size;

    format = other.format;

    return *this;
}

void attachments::resize(size_t size)
{
    this->size = size;
    image.resize(size);
    imageMemory.resize(size);
    imageView.resize(size);
}

void attachments::create(VkPhysicalDevice* physicalDevice, VkDevice* device, VkFormat format, VkImageUsageFlags usage, VkExtent2D extent, uint32_t count)
{
    resize(count);
    this->format = format;
    clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    for(size_t Image=0; Image<count; Image++)
    {
        createImage(        physicalDevice,
                            device,
                            extent.width,
                            extent.height,
                            1,
                            VK_SAMPLE_COUNT_1_BIT,
                            format,
                            VK_IMAGE_TILING_OPTIMAL,
                            usage,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            image[Image],
                            imageMemory[Image]);

        createImageView(    device,
                            image[Image],
                            format,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            1,
                            &imageView[Image]);
    }
}

void attachments::createDepth(VkPhysicalDevice* physicalDevice, VkDevice* device, VkFormat format, VkImageUsageFlags usage, VkExtent2D extent, uint32_t count)
{
    resize(count);
    this->format = format;
    clearValue.depthStencil = {1.0f, 0};
    for(size_t Image=0; Image<count; Image++)
    {
        createImage(        physicalDevice,
                            device,
                            extent.width,
                            extent.height,
                            1,
                            VK_SAMPLE_COUNT_1_BIT,
                            format,
                            VK_IMAGE_TILING_OPTIMAL,
                            usage,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            image[Image],
                            imageMemory[Image]);

        createImageView(    device,
                            image[Image],
                            format,
                            VK_IMAGE_ASPECT_DEPTH_BIT,
                            1,
                            &imageView[Image]);
    }
}

void attachments::deleteAttachment(VkDevice * device)
{
    for(size_t i=0;i<size;i++)
    {
        if(image[i])        vkDestroyImage(*device, image[i], nullptr);
        if(imageMemory[i])  vkFreeMemory(*device, imageMemory[i], nullptr);
        if(imageView[i])    vkDestroyImageView(*device, imageView[i], nullptr);
    }
    image.resize(0);
    imageMemory.resize(0);
    imageView.resize(0);
    size = 0;
}

void attachments::deleteSampler(VkDevice *device)
{
    if(sampler) vkDestroySampler(*device,sampler,nullptr);
}

size_t attachments::getSize()
{
    return size;
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

void DeferredAttachments::deleteAttachment(VkDevice * device){
    image.deleteAttachment(device);
    blur.deleteAttachment(device);
    bloom.deleteAttachment(device);
    depth.deleteAttachment(device);
    GBuffer.color.deleteAttachment(device);
    GBuffer.emission.deleteAttachment(device);
    GBuffer.normal.deleteAttachment(device);
    GBuffer.position.deleteAttachment(device);
}

void DeferredAttachments::deleteSampler(VkDevice * device){
    image.deleteSampler(device);
    blur.deleteSampler(device);
    bloom.deleteSampler(device);
    depth.deleteSampler(device);
    GBuffer.color.deleteSampler(device);
    GBuffer.emission.deleteSampler(device);
    GBuffer.normal.deleteSampler(device);
    GBuffer.position.deleteSampler(device);
}
