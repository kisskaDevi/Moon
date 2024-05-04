#ifndef DEPTHMAP_H
#define DEPTHMAP_H

#include "attachments.h"
#include "device.h"

namespace moon::utils {

class Texture;

class DepthMap {
private:
    Attachments*                    map{nullptr};
    VkDescriptorSetLayout           descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                descriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    descriptorSets;

    Texture*                        emptyTextureBlack{nullptr};
    Texture*                        emptyTextureWhite{nullptr};
    VkDevice                        device{VK_NULL_HANDLE};

    void createDescriptorPool(VkDevice device, uint32_t imageCount);
    void createDescriptorSets(VkDevice device, uint32_t imageCount);
public:
    DepthMap(PhysicalDevice device, VkCommandPool commandPool, uint32_t imageCount);
    ~DepthMap();
    void destroy(VkDevice device);

    const std::vector<VkDescriptorSet>& getDescriptorSets() const;
    void updateDescriptorSets(VkDevice device, uint32_t imageCount);
    Attachments* &get();

    static void createDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
};

}
#endif // DEPTHMAP_H
