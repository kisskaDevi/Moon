#include "attachments.h"
#include "operations.h"
#include "vkdefault.h"
#include <texture.h>
#include <algorithm>
#include <iterator>

namespace moon::utils {

Attachment::Attachment(Attachment&& other) noexcept {
    this->swap(other);
};

Attachment& Attachment::operator=(Attachment&& other) noexcept {
    this->swap(other);
    return *this;
};

void Attachment::swap(Attachment& other) noexcept {
    uint8_t buff[sizeof(Attachment)];
    std::memcpy((void*)buff, (void*)&other, sizeof(Attachment));
    std::memcpy((void*)&other, (void*)this, sizeof(Attachment));
    std::memcpy((void*)this, (void*)buff, sizeof(Attachment));
}

Attachment::Attachment(VkPhysicalDevice physicalDevice, VkDevice device, ImageInfo imageInfo, VkImageUsageFlags usage) {
    this->device = device;
    const auto depthFormats = image::depthFormats();
    const bool isDepth = std::any_of(depthFormats.begin(), depthFormats.end(), [&imageInfo](const VkFormat& format) {return imageInfo.Format == format; });
    const VkImageAspectFlagBits imageAspectFlagBits = isDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    image = utils::vkDefault::Image(physicalDevice, device, 0, { imageInfo.Extent.width, imageInfo.Extent.height, 1 }, 1, 1, VK_SAMPLE_COUNT_1_BIT, imageInfo.Format, VK_IMAGE_LAYOUT_UNDEFINED, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    imageView = utils::vkDefault::ImageView(device, image, VK_IMAGE_VIEW_TYPE_2D, imageInfo.Format, imageAspectFlagBits, 1, 0, 1);
}

Attachments::Attachments(Attachments&& other) noexcept {
    this->swap(other);
}

Attachments& Attachments::operator=(Attachments&& other) noexcept {
    this->swap(other);
    return *this;
}

void Attachments::swap(Attachments& other) noexcept {
    std::swap(instances, other.instances);
    std::swap(imageSampler, other.imageSampler);
    std::swap(imageInfo, other.imageInfo);
    std::swap(imageClearValue, other.imageClearValue);
}

VkResult Attachments::create(VkPhysicalDevice physicalDevice, VkDevice device, ImageInfo imageInfo, VkImageUsageFlags usage, VkClearValue clear, VkSamplerCreateInfo samplerInfo) {
    this->imageInfo = imageInfo;
    imageClearValue = clear;
    VkResult result = VK_SUCCESS;

    instances.resize(imageInfo.Count);
    for(auto& instance : instances){
        instance = Attachment(physicalDevice, device, imageInfo, usage);
        Memory::instance().nameMemory(instance.image, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", attachments::create, instance " + std::to_string(&instance - &instances[0]));
    }
    imageSampler = utils::vkDefault::Sampler(device, samplerInfo);
    return result;
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

void createAttachments(VkPhysicalDevice physicalDevice, VkDevice device, const ImageInfo image, uint32_t attachmentsCount, Attachments* pAttachments, VkImageUsageFlags usage, VkSamplerCreateInfo samplerInfo){
    for(uint32_t i = 0; i < attachmentsCount; i++){
        CHECK(pAttachments[i].create(physicalDevice, device, image, usage, { {0.0f,0.0f,0.0f,0.0f}}, samplerInfo));
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

    return attachment ? attachment->imageView(imageIndex) : emptyTexture->imageView();
}

VkSampler AttachmentsDatabase::sampler(const std::string& id, const std::optional<std::string>& emptyTextureId) const {
    const auto emptyTexture = emptyTextureId ? emptyTexturesMap.at(*emptyTextureId) : emptyTexturesMap.at(defaultEmptyTexture);
    const auto attachment = get(id);

    return attachment ? attachment->sampler() : emptyTexture->sampler();
}

VkDescriptorImageInfo AttachmentsDatabase::descriptorImageInfo(const std::string& id, const uint32_t imageIndex, const std::optional<std::string>& emptyTextureId) const{
    VkDescriptorImageInfo res;
    res.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    res.imageView = imageView(id, imageIndex, emptyTextureId);
    res.sampler = sampler(id);
    return res;
}

VkDescriptorImageInfo AttachmentsDatabase::descriptorEmptyInfo(const std::string& id) const {
    VkDescriptorImageInfo res;
    res.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    res.imageView = getEmpty(id)->imageView();
    res.sampler = getEmpty(id)->sampler();
    return res;
}

}
