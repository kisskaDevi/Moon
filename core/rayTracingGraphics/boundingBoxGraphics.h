#ifndef BOUNDINGBOXGRAPHICS_H
#define BOUNDINGBOXGRAPHICS_H

#include "attachments.h"
#include "buffer.h"
#include "vkdefault.h"

#include "utils/devicep.h"
#include "math/box.h"
#include "transformational/camera.h"

#include <filesystem>

namespace moon::rayTracingGraphics {

class BoundingBoxGraphics
{
private:
    VkPhysicalDevice                physicalDevice{VK_NULL_HANDLE};
    VkDevice                        device{VK_NULL_HANDLE};
    moon::utils::ImageInfo          image;

    std::filesystem::path           vertShaderPath;
    std::filesystem::path           fragShaderPath;

    utils::vkDefault::PipelineLayout        pipelineLayout;
    utils::vkDefault::Pipeline              pipeline;
    utils::vkDefault::DescriptorSetLayout   descriptorSetLayout;
    utils::vkDefault::DescriptorPool        descriptorPool;
    utils::vkDefault::DescriptorSets        descriptorSets;
    utils::vkDefault::RenderPass            renderPass;
    utils::vkDefault::Framebuffers          framebuffers;

    moon::utils::Attachments frame;
    bool enable{true};

    std::vector<cuda::rayTracing::cbox> boxes;
    cuda::rayTracing::Devicep<cuda::rayTracing::Camera>* camera;
    moon::utils::Buffers cameraBuffers;

    void createAttachments();
    void createRenderPass();
    void createFramebuffers();
    void createDescriptorSetLayout();
    void createPipeline();
    void createDescriptors();

public:
    BoundingBoxGraphics() = default;

    void create(VkPhysicalDevice physicalDevice, VkDevice device, const moon::utils::ImageInfo& image, const std::filesystem::path& shadersPath);
    void update(uint32_t imageIndex);
    void render(VkCommandBuffer commandBuffer, uint32_t imageIndex) const;

    const moon::utils::Attachments& getAttachments() const;
    void clear();
    void bind(const cuda::rayTracing::cbox& box);
    void bind(cuda::rayTracing::Devicep<cuda::rayTracing::Camera>* camera);

    void setEnable(bool enable){this->enable = enable;}
    bool getEnable(){return enable;}
};

}
#endif // BOUNDINGBOXGRAPHICS_H
