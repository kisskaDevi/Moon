#include "attachments.h"

attachments::attachments()
{

}

attachments::~attachments()
{

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
