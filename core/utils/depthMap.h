#ifndef DEPTHMAP_H
#define DEPTHMAP_H

#include "attachments.h"
#include "device.h"
#include "texture.h"

namespace moon::utils {

class DepthMap {
private:
    Attachments*                    map{nullptr};
    moon::utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool                descriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    descriptorSets;

    Texture                         emptyTextureWhite;
    VkDevice                        device{VK_NULL_HANDLE};

    void createDescriptorPool(VkDevice device, uint32_t imageCount);
    void createDescriptorSets(VkDevice device, uint32_t imageCount);
public:
    DepthMap(const PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount);
    ~DepthMap();
    void destroy(VkDevice device);

    const std::vector<VkDescriptorSet>& getDescriptorSets() const;
    void updateDescriptorSets(VkDevice device, uint32_t imageCount);
    Attachments* &get();

    static moon::utils::vkDefault::DescriptorSetLayout createDescriptorSetLayout(VkDevice device);
};

}
#endif // DEPTHMAP_H
