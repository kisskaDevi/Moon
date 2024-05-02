#ifndef BOUNDINGBOXGRAPHICS_H
#define BOUNDINGBOXGRAPHICS_H

#include "attachments.h"
#include "buffer.h"

#include "hitable/hitable.h"
#include "transformational/camera.h"

#include <filesystem>

class boundingBoxGraphics
{
private:
    VkPhysicalDevice                physicalDevice{VK_NULL_HANDLE};
    VkDevice                        device{VK_NULL_HANDLE};
    imageInfo                       image;

    std::filesystem::path           vertShaderPath;
    std::filesystem::path           fragShaderPath;

    VkPipelineLayout                pipelineLayout{VK_NULL_HANDLE};
    VkPipeline                      pipeline{VK_NULL_HANDLE};
    VkDescriptorSetLayout           descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                descriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    descriptorSets;

    VkRenderPass                    renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>      framebuffers;

    attachments frame;
    bool enable{true};

    std::vector<cuda::cbox> boxes;
    cuda::devicep<cuda::camera>* camera;
    buffers cameraBuffers;

    void createAttachments();
    void createRenderPass();
    void createFramebuffers();

    void createDescriptorSetLayout();
    void createPipeline();

    void createDescriptorPool();
    void createDescriptorSets();

public:
    boundingBoxGraphics();
    ~boundingBoxGraphics();

    void destroy();
    void create(VkPhysicalDevice physicalDevice, VkDevice device, const imageInfo& image, const std::filesystem::path& shadersPath);
    void update(uint32_t imageIndex);
    void render(VkCommandBuffer commandBuffer, uint32_t imageIndex) const;

    const attachments& getAttachments() const;
    void bind(cuda::cbox box);
    void bind(cuda::devicep<cuda::camera>* camera);

    void setEnable(bool enable){this->enable = enable;}
    bool getEnable(){return enable;}
};

#endif // BOUNDINGBOXGRAPHICS_H
