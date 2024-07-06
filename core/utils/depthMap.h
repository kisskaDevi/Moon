#ifndef DEPTHMAP_H
#define DEPTHMAP_H

#include "attachments.h"
#include "device.h"
#include "texture.h"

namespace moon::utils {

class DepthMap {
private:
    struct {
        utils::Attachments attachments;
        utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
        utils::vkDefault::DescriptorPool descriptorPool;
        utils::vkDefault::DescriptorSets descriptorSets;
    } map;

    Texture emptyTextureWhite;
    utils::ImageInfo imageInfo;
    VkDevice device{VK_NULL_HANDLE};

public:
    DepthMap() = default;
    DepthMap(const PhysicalDevice& device, VkCommandPool commandPool, const utils::ImageInfo& imageInfo);
    void update(bool enable);

    const utils::vkDefault::DescriptorSets& descriptorSets() const;
    const utils::Attachments& attachments() const;

    static moon::utils::vkDefault::DescriptorSetLayout createDescriptorSetLayout(VkDevice device);
};

}
#endif // DEPTHMAP_H
