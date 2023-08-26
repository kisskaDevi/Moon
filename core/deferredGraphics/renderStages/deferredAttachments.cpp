#include "deferredAttachments.h"


GBufferAttachments::GBufferAttachments()
{}

GBufferAttachments::GBufferAttachments(const GBufferAttachments& other):
      position(other.position),
      normal(other.normal),
      color(other.color),
      depth(other.depth)
{}

GBufferAttachments& GBufferAttachments::operator=(const GBufferAttachments& other)
{
    position = other.position;
    normal = other.normal;
    color = other.color;
    depth = other.depth;
    return *this;
}

DeferredAttachments::DeferredAttachments()
{}

DeferredAttachments::DeferredAttachments(const DeferredAttachments& other):
    image(other.image),
    blur(other.blur),
    bloom(other.bloom),
    GBuffer(other.GBuffer)
{}

DeferredAttachments& DeferredAttachments::operator=(const DeferredAttachments& other)
{
    image = other.image;
    blur = other.blur;
    bloom = other.bloom;
    GBuffer = other.GBuffer;
    return *this;
}

void DeferredAttachments::deleteAttachment(VkDevice device){
    image.deleteAttachment(device);
    blur.deleteAttachment(device);
    bloom.deleteAttachment(device);
    GBuffer.color.deleteAttachment(device);
    GBuffer.normal.deleteAttachment(device);
    GBuffer.position.deleteAttachment(device);
    GBuffer.depth.deleteAttachment(device);
}

void DeferredAttachments::deleteSampler(VkDevice device){
    image.deleteSampler(device);
    blur.deleteSampler(device);
    bloom.deleteSampler(device);
    GBuffer.color.deleteSampler(device);
    GBuffer.normal.deleteSampler(device);
    GBuffer.position.deleteSampler(device);
    GBuffer.depth.deleteSampler(device);
}
