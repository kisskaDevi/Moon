#ifndef ATTACHMENTS_H
#define ATTACHMENTS_H

#include <vulkan.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <optional>

#include <vkdefault.h>

namespace moon::utils {

struct ImageInfo{
    uint32_t                Count{0};
    VkFormat                Format{ VK_FORMAT_UNDEFINED };
    VkExtent2D              Extent{0, 0};
    VkSampleCountFlagBits   Samples{ VK_SAMPLE_COUNT_1_BIT };
};

struct Attachment{
    VkImage         image{VK_NULL_HANDLE};
    VkDeviceMemory  imageMemory{VK_NULL_HANDLE};
    VkImageView     imageView{VK_NULL_HANDLE};
    VkImageLayout   layout{VK_IMAGE_LAYOUT_UNDEFINED};
    VkDevice        device{ VK_NULL_HANDLE };

    virtual ~Attachment();
};

class Attachments{
private:
    std::vector<Attachment> instances;
    VkSampler imageSampler{ VK_NULL_HANDLE };
    ImageInfo imageInfo;
    VkClearValue imageClearValue{};

    VkDevice device{ VK_NULL_HANDLE };

    void destroy();
public:
    Attachments() = default;
    Attachments(const Attachments& other) = delete;
    Attachments& operator=(const Attachments& other) = delete;
    Attachments(Attachments&& other){
        instances = std::move(other.instances);
        imageSampler = std::move(other.imageSampler);
        imageInfo = std::move(other.imageInfo);
        imageClearValue = std::move(other.imageClearValue);
    };
    Attachments& operator=(Attachments&& other) {
        destroy();
        instances = std::move(other.instances);
        imageSampler = std::move(other.imageSampler);
        imageInfo = std::move(other.imageInfo);
        imageClearValue = std::move(other.imageClearValue);
        return *this;
    }

    ~Attachments();

    VkResult create(VkPhysicalDevice physicalDevice, VkDevice device, ImageInfo imageInfo, VkImageUsageFlags usage, VkClearValue clear = {{0.0f, 0.0f, 0.0f, 0.0f}}, VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO });
    VkResult createDepth(VkPhysicalDevice physicalDevice, VkDevice device, ImageInfo imageInfo, VkImageUsageFlags usage, VkClearValue clear = {{ 1.0f, 0 }}, VkSamplerCreateInfo samplerInfo = vkDefault::samler());

    static VkAttachmentDescription imageDescription(VkFormat format);
    static VkAttachmentDescription imageDescription(VkFormat format, VkImageLayout layout);
    static VkAttachmentDescription depthStencilDescription(VkFormat format);
    static VkAttachmentDescription depthDescription(VkFormat format);

    std::vector<VkImage> getImages() const;

    const VkImage& image(size_t i) const { return instances[i].image; }
    const VkImageView& imageView(size_t i) const { return instances[i].imageView; }
    const VkSampler& sampler() const {return imageSampler;}
    const VkFormat& format() const { return imageInfo.Format; }
    const uint32_t& count() const { return imageInfo.Count; }
    const VkClearValue& clearValue() const { return imageClearValue; }

    Attachment& attachment(size_t i) { return instances[i];}
};

void createAttachments(VkPhysicalDevice physicalDevice, VkDevice device, const ImageInfo image, uint32_t attachmentsCount, Attachments* pAttachments, VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO });

class Texture;

struct AttachmentsDatabase{
    struct data{
        bool enable{false};
        const Attachments* pImages{nullptr};
    };

    std::string defaultEmptyTexture;
    std::unordered_map<std::string, Texture*> emptyTexturesMap;
    std::unordered_map<std::string, data> attachmentsMap;

    AttachmentsDatabase() = default;
    AttachmentsDatabase(const std::string& emptyTextureId, Texture* emptyTexture);
    AttachmentsDatabase(const AttachmentsDatabase&) = default;
    AttachmentsDatabase& operator=(const AttachmentsDatabase&) = default;

    void destroy();

    bool addEmptyTexture(const std::string& id, Texture* emptyTexture);
    bool addAttachmentData(const std::string& id, bool enable, const Attachments* pImages);
    bool enable(const std::string& id) const;
    const Attachments* get(const std::string& id) const;
    const Texture* getEmpty(const std::string& id = "") const;
    VkImageView imageView(const std::string& id, const uint32_t imageIndex, const std::optional<std::string>& emptyTextureId = std::nullopt) const;
    VkSampler sampler(const std::string& id, const std::optional<std::string>& emptyTextureId = std::nullopt) const;
    VkDescriptorImageInfo descriptorImageInfo(const std::string& id, const uint32_t imageIndex, const std::optional<std::string>& emptyTextureId = std::nullopt) const;
};

}
#endif // ATTACHMENTS_H
