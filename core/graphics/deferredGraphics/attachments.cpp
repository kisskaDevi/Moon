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
        vkDestroyImage(*device, image.at(i), nullptr);
        vkFreeMemory(*device, imageMemory.at(i), nullptr);
        vkDestroyImageView(*device, imageView.at(i), nullptr);
    }
}

void attachments::deleteSampler(VkDevice *device)
{
    vkDestroySampler(*device,sampler,nullptr);
}

void attachments::setSize(size_t size)
{
    this->size = size;
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

