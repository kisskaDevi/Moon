#include "attachments.h"

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

    return *this;
}

void attachments::resize(size_t size)
{
    this->size = size;
    image.resize(size);
    imageMemory.resize(size);
    imageView.resize(size);
}

void attachments::deleteAttachment(VkDevice * device)
{
    for(size_t i=0;i<size;i++)
    {
        if(image[i])        vkDestroyImage(*device, image[i], nullptr);
        if(imageMemory[i])  vkFreeMemory(*device, imageMemory[i], nullptr);
        if(imageView[i])    vkDestroyImageView(*device, imageView[i], nullptr);
    }
}

void attachments::deleteSampler(VkDevice *device)
{
    if(sampler) vkDestroySampler(*device,sampler,nullptr);
}

size_t attachments::getSize()
{
    return size;
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
