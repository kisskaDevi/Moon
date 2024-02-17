#ifndef DEPTHMAP_H
#define DEPTHMAP_H

#include "attachments.h"
#include "device.h"

class texture;

class depthMap {
private:
    attachments*                    map{nullptr};
    VkDescriptorSetLayout           descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                descriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    descriptorSets;

    texture*                        emptyTextureBlack{nullptr};
    texture*                        emptyTextureWhite{nullptr};
    VkDevice                        device{VK_NULL_HANDLE};

    void createDescriptorPool(VkDevice device, uint32_t imageCount);
    void createDescriptorSets(VkDevice device, uint32_t imageCount);
public:
    depthMap(physicalDevice device, VkCommandPool commandPool, uint32_t imageCount);
    ~depthMap();
    void destroy(VkDevice device);

    const std::vector<VkDescriptorSet>& getDescriptorSets() const;
    void updateDescriptorSets(VkDevice device, uint32_t imageCount);
    attachments* &get();

    static void createDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
};

#endif // DEPTHMAP_H
