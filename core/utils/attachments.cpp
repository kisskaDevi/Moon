#include "attachments.h"
#include "operations.h"
#include "vkdefault.h"
#include <algorithm>
#include <iterator>

attachments::attachments(const attachments &other)
{
    std::copy(other.instances.begin(), other.instances.end(), std::back_inserter(instances));
    sampler = other.sampler;
    format = other.format;
}

attachments& attachments::operator=(const attachments& other)
{
    std::copy(other.instances.begin(), other.instances.end(), std::back_inserter(instances));
    sampler = other.sampler;
    format = other.format;

    return *this;
}

VkResult attachments::create(VkPhysicalDevice physicalDevice, VkDevice device, VkFormat format, VkImageUsageFlags usage, VkExtent2D extent, uint32_t count)
{
    VkResult result = VK_SUCCESS;

    instances.resize(count);

    this->format = format;
    clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

    for(auto& instance : instances){
        result = Texture::create(   physicalDevice,
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
                                    &(instance.image),
                                    &(instance.imageMemory));
        CHECK(result);

        Memory::instance().nameMemory(instance.imageMemory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", attachments::create, instance " + std::to_string(&instance - &instances[0]));

        result = Texture::createView(   device,
                                        VK_IMAGE_VIEW_TYPE_2D,
                                        format,
                                        VK_IMAGE_ASPECT_COLOR_BIT,
                                        1,
                                        0,
                                        1,
                                        instance.image,
                                        &(instance.imageView));
        CHECK(result);
    }
    return result;
}

VkResult attachments::createDepth(VkPhysicalDevice physicalDevice, VkDevice device, VkFormat format, VkImageUsageFlags usage, VkExtent2D extent, uint32_t count)
{
    VkResult result = VK_SUCCESS;

    instances.resize(count);

    this->format = format;
    clearValue.depthStencil = {1.0f, 0};

    for(auto& instance : instances){
        result = Texture::create(   physicalDevice,
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
                                    &(instance.image),
                                    &(instance.imageMemory));
        CHECK(result);

        Memory::instance().nameMemory(instance.imageMemory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", attachments::createDepth, instance " + std::to_string(&instance - &instances[0]));

        result = Texture::createView(   device,
                                        VK_IMAGE_VIEW_TYPE_2D,
                                        format,
                                        VK_IMAGE_ASPECT_DEPTH_BIT,
                                        1,
                                        0,
                                        1,
                                        instance.image,
                                        &(instance.imageView));
        CHECK(result);
    }
    return result;
}

void attachments::deleteAttachment(VkDevice device)
{
    std::for_each(instances.begin(), instances.end(), [&device](attachment& instance){
        Texture::destroy(device, instance.image, instance.imageMemory);
        vkDestroyImageView(device, instance.imageView, nullptr);
        instance.imageView = VK_NULL_HANDLE;
    });
    instances.clear();
}

void attachments::deleteSampler(VkDevice device)
{
    if(sampler){
        vkDestroySampler(device,sampler,nullptr);
        sampler = VK_NULL_HANDLE;
    }
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

std::vector<VkImage> attachments::getImages() const {
    std::vector<VkImage> images;
    for (auto& instance: instances){
        images.push_back(instance.image);
    }
    return images;
}

void createAttachments(VkPhysicalDevice physicalDevice, VkDevice device, const imageInfo image, uint32_t attachmentsCount, attachments* pAttachments, VkImageUsageFlags usage){
    for(VkSamplerCreateInfo samplerInfo = vkDefault::samler(); 0 < attachmentsCount; attachmentsCount--){
        pAttachments[attachmentsCount - 1].create(physicalDevice,device,image.Format,usage,image.frameBufferExtent,image.Count);
        CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &pAttachments[attachmentsCount - 1].sampler));
        pAttachments->clearValue.color = {{0.0f,0.0f,0.0f,1.0f}};
    }
}
