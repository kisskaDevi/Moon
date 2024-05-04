#include "attachments.h"
#include "operations.h"
#include "vkdefault.h"
#include <texture.h>
#include <algorithm>
#include <iterator>

namespace moon::utils {

Attachments::Attachments(const Attachments &other){
    std::copy(other.instances.begin(), other.instances.end(), std::back_inserter(instances));
    sampler = other.sampler;
    format = other.format;
}

Attachments& Attachments::operator=(const Attachments& other){
    std::copy(other.instances.begin(), other.instances.end(), std::back_inserter(instances));
    sampler = other.sampler;
    format = other.format;

    return *this;
}

VkResult Attachments::create(VkPhysicalDevice physicalDevice, VkDevice device, VkFormat format, VkImageUsageFlags usage, VkExtent2D extent, uint32_t count)
{
    VkResult result = VK_SUCCESS;

    instances.resize(count);

    this->format = format;
    clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

    for(auto& instance : instances){
        result = texture::create(   physicalDevice,
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

        result = texture::createView(   device,
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

VkResult Attachments::createDepth(VkPhysicalDevice physicalDevice, VkDevice device, VkFormat format, VkImageUsageFlags usage, VkExtent2D extent, uint32_t count)
{
    VkResult result = VK_SUCCESS;

    instances.resize(count);

    this->format = format;
    clearValue.depthStencil = {1.0f, 0};

    for(auto& instance : instances){
        result = texture::create(   physicalDevice,
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

        result = texture::createView(   device,
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

void Attachments::deleteAttachment(VkDevice device)
{
    std::for_each(instances.begin(), instances.end(), [&device](Attachment& instance){
        texture::destroy(device, instance.image, instance.imageMemory);
        vkDestroyImageView(device, instance.imageView, nullptr);
        instance.imageView = VK_NULL_HANDLE;
    });
    instances.clear();
}

void Attachments::deleteSampler(VkDevice device)
{
    if(sampler){
        vkDestroySampler(device,sampler,nullptr);
        sampler = VK_NULL_HANDLE;
    }
}

VkAttachmentDescription Attachments::imageDescription(VkFormat format)
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

VkAttachmentDescription Attachments::imageDescription(VkFormat format, VkImageLayout layout)
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

VkAttachmentDescription Attachments::depthDescription(VkFormat format)
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

VkAttachmentDescription Attachments::depthStencilDescription(VkFormat format)
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

std::vector<VkImage> Attachments::getImages() const {
    std::vector<VkImage> images;
    for (const auto& instance: instances){
        images.push_back(instance.image);
    }
    return images;
}

void createAttachments(VkPhysicalDevice physicalDevice, VkDevice device, const ImageInfo image, uint32_t attachmentsCount, Attachments* pAttachments, VkImageUsageFlags usage){
    for(VkSamplerCreateInfo samplerInfo = vkDefault::samler(); 0 < attachmentsCount; attachmentsCount--){
        pAttachments[attachmentsCount - 1].create(physicalDevice,device,image.Format,usage,image.Extent,image.Count);
        CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &pAttachments[attachmentsCount - 1].sampler));
        pAttachments->clearValue.color = {{0.0f,0.0f,0.0f,1.0f}};
    }
}

AttachmentsDatabase::AttachmentsDatabase(const std::string& emptyTextureId, Texture* emptyTexture)
{
    defaultEmptyTexture = emptyTextureId;
    emptyTexturesMap[emptyTextureId] = emptyTexture;
}

void AttachmentsDatabase::destroy(){
    attachmentsMap.clear();
    emptyTexturesMap.clear();
    defaultEmptyTexture.clear();
}

bool AttachmentsDatabase::addEmptyTexture(const std::string& id, Texture* emptyTexture){
    if(emptyTexturesMap.count(id) > 0) return false;
    if(defaultEmptyTexture.empty()) defaultEmptyTexture = id;

    emptyTexturesMap[id] = emptyTexture;
    return true;
}

bool AttachmentsDatabase::addAttachmentData(const std::string& id, bool enable, const Attachments* pImages){
    if(attachmentsMap.count(id) > 0) return false;

    attachmentsMap[id] = data{enable, pImages};
    return true;
}

bool AttachmentsDatabase::enable(const std::string& id) const {
    return attachmentsMap.at(id).enable;
}

const Attachments* AttachmentsDatabase::get(const std::string& id) const{
    return attachmentsMap.count(id) > 0 && attachmentsMap.at(id).enable ? attachmentsMap.at(id).pImages : nullptr;
}

const Texture* AttachmentsDatabase::getEmpty(const std::string& id) const {
    const auto texid = id.empty() ? defaultEmptyTexture : id;
    return emptyTexturesMap.count(texid) > 0 ? emptyTexturesMap.at(texid) : nullptr;
}

VkImageView AttachmentsDatabase::imageView(const std::string& id, const uint32_t imageIndex, const std::optional<std::string>& emptyTextureId) const {
    const auto emptyTexture = emptyTextureId ? emptyTexturesMap.at(*emptyTextureId) : emptyTexturesMap.at(defaultEmptyTexture);
    const auto attachment = get(id);

    return attachment ? attachment->instances[imageIndex].imageView : *emptyTexture->getTextureImageView();
}

VkSampler AttachmentsDatabase::sampler(const std::string& id, const std::optional<std::string>& emptyTextureId) const {
    const auto emptyTexture = emptyTextureId ? emptyTexturesMap.at(*emptyTextureId) : emptyTexturesMap.at(defaultEmptyTexture);
    const auto attachment = get(id);

    return attachment ? attachment->sampler : *emptyTexture->getTextureSampler();
}

VkDescriptorImageInfo AttachmentsDatabase::descriptorImageInfo(const std::string& id, const uint32_t imageIndex, const std::optional<std::string>& emptyTextureId) const{
    VkDescriptorImageInfo res;
    res.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    res.imageView = imageView(id, imageIndex, emptyTextureId);
    res.sampler = sampler(id);
    return res;
}

}
