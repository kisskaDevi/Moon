#ifndef LAYERSCOMBINER_H
#define LAYERSCOMBINER_H

#include "filtergraphics.h"

class camera;

class layersCombiner : public filterGraphics
{
private:
    VkPhysicalDevice*                   physicalDevice{nullptr};
    VkDevice*                           device{nullptr};

    texture*                            emptyTexture{nullptr};

    imageInfo                           image;

    uint32_t                            attachmentsCount{0};
    attachments*                        pAttachments{nullptr};

    VkRenderPass                        renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>          framebuffers;

    std::vector<VkCommandBuffer>          commandBuffers;

    struct Combiner : public filter{
        std::string                     ExternalPath;
        uint32_t                        transparentLayersCount{0};

        VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
        VkPipeline                      Pipeline{VK_NULL_HANDLE};
        VkDescriptorSetLayout           DescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>    DescriptorSets;
        void Destroy(VkDevice* device) override;
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass) override;
        void createDescriptorSetLayout(VkDevice* device) override;
    }combiner;

public:
    layersCombiner();
    void destroy() override;
    void freeCommandBuffer(VkCommandPool commandPool){
        if(commandBuffers.data()){
            vkFreeCommandBuffers(*device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
        }
        commandBuffers.resize(0);
    }

    void setEmptyTexture(texture* emptyTexture) override;
    void setExternalPath(const std::string& path) override;
    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device) override;
    void setImageProp(imageInfo* pInfo) override;

    void setAttachments(uint32_t attachmentsCount, attachments* pAttachments) override;
    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments) override;
    void createRenderPass() override;
    void createFramebuffers() override;
    void createPipelines() override;

    void createDescriptorPool() override;
    void createDescriptorSets() override;
    void updateDescriptorSets(DeferredAttachments deferredAttachments, DeferredAttachments* transparencyLayers, attachments* skybox, camera* cameraObject);

    void createCommandBuffers(VkCommandPool commandPool) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
    VkCommandBuffer& getCommandBuffer(uint32_t frameNumber) override;

    void setTransparentLayersCount(uint32_t transparentLayersCount);
};

#endif // LAYERSCOMBINER_H
