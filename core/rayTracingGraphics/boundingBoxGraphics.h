#ifndef BOUNDINGBOXGRAPHICS_H
#define BOUNDINGBOXGRAPHICS_H

#include "attachments.h"
#include "buffer.h"

#include "utils/devicep.h"
#include "math/box.h"
#include "transformational/camera.h"

#include <filesystem>

class boundingBoxGraphics
{
private:
    VkPhysicalDevice                physicalDevice{VK_NULL_HANDLE};
    VkDevice                        device{VK_NULL_HANDLE};
    moon::utils::ImageInfo          image;

    std::filesystem::path           vertShaderPath;
    std::filesystem::path           fragShaderPath;

    VkPipelineLayout                pipelineLayout{VK_NULL_HANDLE};
    VkPipeline                      pipeline{VK_NULL_HANDLE};
    VkDescriptorSetLayout           descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                descriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    descriptorSets;

    VkRenderPass                    renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>      framebuffers;

    moon::utils::Attachments frame;
    bool enable{true};

    std::vector<cuda::cbox> boxes;
    cuda::devicep<cuda::camera>* camera;
    moon::utils::Buffers cameraBuffers;

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
    void create(VkPhysicalDevice physicalDevice, VkDevice device, const moon::utils::ImageInfo& image, const std::filesystem::path& shadersPath);
    void update(uint32_t imageIndex);
    void render(VkCommandBuffer commandBuffer, uint32_t imageIndex) const;

    const moon::utils::Attachments& getAttachments() const;
    void clear();
    void bind(const cuda::cbox& box);
    void bind(cuda::devicep<cuda::camera>* camera);

    void setEnable(bool enable){this->enable = enable;}
    bool getEnable(){return enable;}
};

#endif // BOUNDINGBOXGRAPHICS_H
