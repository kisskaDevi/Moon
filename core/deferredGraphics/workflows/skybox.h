#ifndef SKYBOX_H
#define SKYBOX_H

#include "workflow.h"

class object;

struct skyboxAttachments
{
    attachments color;
    attachments bloom;

    inline uint32_t size() const{
        return 2;
    }
    inline attachments* operator&(){
        return &color;
    }
    void deleteAttachment(VkDevice device){
        color.deleteAttachment(device);
        bloom.deleteAttachment(device);
    }
    void deleteSampler(VkDevice device){
        color.deleteSampler(device);
        bloom.deleteSampler(device);
    }
};

class skyboxGraphics : public workflow
{
private:
    skyboxAttachments frame;
    bool enable;

    struct Skybox : public workbody{
        VkDescriptorSetLayout   ObjectDescriptorSetLayout{VK_NULL_HANDLE};

        std::vector<object*>*   objects;

        void destroy(VkDevice device);
        void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) override;
        void createDescriptorSetLayout(VkDevice device) override;
    }skybox;

    void createAttachments(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    skyboxGraphics(bool enable, std::vector<object*>* object = nullptr);

    void destroy() override;
    void create(std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap) override;
    void updateDescriptorSets(
        const std::unordered_map<std::string, std::pair<VkDeviceSize,std::vector<VkBuffer>>>& bufferMap,
        const std::unordered_map<std::string, std::pair<bool,std::vector<attachments*>>>& attachmentsMap) override;
    void updateCommandBuffer(uint32_t frameNumber) override;
};

#endif // SKYBOX_H
