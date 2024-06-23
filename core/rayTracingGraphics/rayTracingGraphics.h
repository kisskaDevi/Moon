#ifndef RAYTRACINGGRAPHICS
#define RAYTRACINGGRAPHICS

#include "rayTracingLink.h"
#include "graphicsInterface.h"
#include "attachments.h"
#include "buffer.h"
#include "texture.h"
#include "vector.h"

#include <stdint.h>
#include <random>
#include <stack>

#include "cudaRayTracing.h"
#include "boundingBoxGraphics.h"
#include <bloom.h>

namespace cuda::rayTracing { struct Object;}

namespace moon::rayTracingGraphics {

class RayTracingGraphics : public moon::graphicsManager::GraphicsInterface {
private:
    struct ImageResource{
        std::string id;
        uint32_t* host{nullptr};
        moon::utils::Buffer hostDevice;
        moon::utils::Attachments device;

        void create(const std::string& id, const moon::utils::PhysicalDevice& phDevice, const moon::utils::ImageInfo& imageInfo);
        void destroy(const moon::utils::PhysicalDevice& phDevice);
        void moveFromHostToHostDevice(VkExtent2D extent);
        void copyToDevice(VkCommandBuffer commandBuffer, VkExtent2D extent, uint32_t imageIndex);
    };

    ImageResource color;
    ImageResource bloom;

    cuda::rayTracing::RayTracing rayTracer;
    RayTracingLink Link;

    BoundingBoxGraphics bbGraphics;
    moon::workflows::BloomGraphics bloomGraph;

    moon::utils::Texture emptyTexture;

    std::filesystem::path shadersPath;
    std::filesystem::path workflowsShadersPath;
    VkExtent2D extent;

    moon::utils::AttachmentsDatabase    aDatabase;
    moon::utils::BuffersDatabase        bDatabase;

    VkCommandPool commandPool{VK_NULL_HANDLE};

    bool bloomEnable = true;

public:
    RayTracingGraphics(const std::filesystem::path& shadersPath, const std::filesystem::path& workflowsShadersPath, VkExtent2D extent);
    ~RayTracingGraphics();

    void setPositionInWindow(const moon::math::Vector<float,2>& offset, const moon::math::Vector<float,2>& size) override;
    void create() override;
    void destroy() override;
    void update(uint32_t imageIndex) override;
    std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) override;

    void setEnableBoundingBox(bool enable);
    void setEnableBloom(bool enable);
    void setBlitFactor(const float& blitFactor);
    void setExtent(VkExtent2D extent);

    void setCamera(cuda::rayTracing::Devicep<cuda::rayTracing::Camera>* cam);
    void bind(cuda::rayTracing::Object* obj);

    void clearFrame();
    void buildTree();
    void buildBoundingBoxes(bool primitive, bool tree, bool onlyLeafs);
};

}
#endif // !RAYTRACINGGRAPHICS

