#ifndef FILTERGRAPHICS_H
#define FILTERGRAPHICS_H

#include <libs/vulkan/vulkan.h>
#include "../attachments.h"

#include <string>

class texture;

class filter{
public:
    std::string                     vertShaderPath;
    std::string                     fragShaderPath;

    VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
    VkPipeline                      Pipeline{VK_NULL_HANDLE};
    VkDescriptorSetLayout           DescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    DescriptorSets;

    void destroy(VkDevice device){
        if(Pipeline)            {vkDestroyPipeline(device, Pipeline, nullptr); Pipeline = VK_NULL_HANDLE;}
        if(PipelineLayout)      {vkDestroyPipelineLayout(device, PipelineLayout,nullptr); PipelineLayout = VK_NULL_HANDLE;}
        if(DescriptorSetLayout) {vkDestroyDescriptorSetLayout(device, DescriptorSetLayout, nullptr); DescriptorSetLayout = VK_NULL_HANDLE;}
        if(DescriptorPool)      {vkDestroyDescriptorPool(device, DescriptorPool, nullptr); DescriptorPool = VK_NULL_HANDLE;}
        DescriptorSets.clear();
    }
    virtual void createPipeline(VkDevice device, imageInfo* pInfo, VkRenderPass pRenderPass) = 0;
    virtual void createDescriptorSetLayout(VkDevice device) = 0;
};

class filterGraphics
{
protected:
    VkPhysicalDevice    physicalDevice{VK_NULL_HANDLE};
    VkDevice            device{VK_NULL_HANDLE};
    std::string         externalPath{};
    imageInfo           image;
    texture*            emptyTexture;

    uint32_t            attachmentsCount{0};
    attachments*        pAttachments{nullptr};

    VkRenderPass                        renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>          framebuffers;

    std::vector<VkCommandBuffer> commandBuffers;
public:
    virtual ~filterGraphics(){};
    void destroy(){
        if(renderPass) {vkDestroyRenderPass(device, renderPass, nullptr); renderPass = VK_NULL_HANDLE;}
        for(size_t i = 0; i< framebuffers.size();i++)
            if(framebuffers[i]) vkDestroyFramebuffer(device, framebuffers[i],nullptr);
        framebuffers.resize(0);
    }

    void setEmptyTexture(texture* emptyTexture){
        this->emptyTexture = emptyTexture;
    }
    void setExternalPath(const std::string &path){
        externalPath = path;
    }
    void setDeviceProp(VkPhysicalDevice physicalDevice, VkDevice device){
        this->physicalDevice = physicalDevice;
        this->device = device;
    }
    void setImageProp(imageInfo* pInfo){
        this->image = *pInfo;
    }
    void setAttachments(uint32_t attachmentsCount, attachments* pAttachments){
        this->attachmentsCount = attachmentsCount;
        this->pAttachments = pAttachments;
    }

    virtual void createAttachments(uint32_t attachmentsCount, attachments* pAttachments) = 0;
    virtual void createRenderPass() = 0;
    virtual void createFramebuffers() = 0;
    virtual void createPipelines() = 0;

    virtual void createDescriptorPool() = 0;
    virtual void createDescriptorSets() = 0;

    virtual void updateCommandBuffer(uint32_t frameNumber) = 0;

    void createCommandBuffers(VkCommandPool commandPool)
    {
        commandBuffers.resize(image.Count);
        VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = static_cast<uint32_t>(image.Count);
        vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());
    }

    void beginCommandBuffer(uint32_t frameNumber){
        vkResetCommandBuffer(commandBuffers[frameNumber],0);

        VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = 0;
            beginInfo.pInheritanceInfo = nullptr;

        vkBeginCommandBuffer(commandBuffers[frameNumber], &beginInfo);
    }

    void endCommandBuffer(uint32_t frameNumber){
        vkEndCommandBuffer(commandBuffers[frameNumber]);
    }

    VkCommandBuffer& getCommandBuffer(uint32_t frameNumber)
    {
        return commandBuffers[frameNumber];
    }

    void freeCommandBuffer(VkCommandPool commandPool){
        if(commandBuffers.data()){
            vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
        }
        commandBuffers.resize(0);
    }

    static void createDescriptorPool(VkDevice device, filter* filter, const uint32_t& bufferCount, const uint32_t& imageCount, const uint32_t& maxSets){
        std::vector<VkDescriptorPoolSize> poolSizes;
        if(bufferCount){
            poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, bufferCount});
        }
        if(imageCount){
            poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageCount});
        }
        VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
            poolInfo.pPoolSizes = poolSizes.data();
            poolInfo.maxSets = maxSets;
        vkCreateDescriptorPool(device, &poolInfo, nullptr, &filter->DescriptorPool);
    }

    static void createDescriptorSets(VkDevice device, filter* filter, const uint32_t& imageCount){
        filter->DescriptorSets.resize(imageCount);
        std::vector<VkDescriptorSetLayout> layouts(imageCount, filter->DescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = filter->DescriptorPool;
            allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
            allocInfo.pSetLayouts = layouts.data();
        vkAllocateDescriptorSets(device, &allocInfo, filter->DescriptorSets.data());
    };
};

#endif // FILTERGRAPHICS_H
