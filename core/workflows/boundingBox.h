#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include "workflow.h"

class object;
class camera;

class boundingBoxGraphics : public workflow
{
private:
    attachments frame;
    bool enable{true};

    struct boundingBox : workbody{
        VkDescriptorSetLayout   ObjectDescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorSetLayout   PrimitiveDescriptorSetLayout{VK_NULL_HANDLE};

        std::vector<object*>    objects;

        void destroy(VkDevice device);
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }box;

    void createAttachments(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    boundingBoxGraphics(bool enable);

    void destroy() override;
    void create(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap) override;
    void updateDescriptorSets(
        const std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>>& bufferMap,
        const std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap) override;
    void updateCommandBuffer(uint32_t frameNumber) override;

    void bindObject(object* object);
    bool removeObject(object* object);
};

#endif // BOUNDINGBOX_H
