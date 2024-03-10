#ifndef RAYTRACINGGRAPHICS
#define RAYTRACINGGRAPHICS

#include "rayTracingLink.h"
#include "graphicsInterface.h"
#include "attachments.h"
#include "buffer.h"
#include "vector.h"

#include <stdint.h>

#include "cudaRayTracing.h"

namespace cuda {
class model;
}

class rayTracingGraphics : public graphicsInterface {
private:
    uint32_t* hostFrameBuffer{nullptr};

    cuda::cudaRayTracing rayTracer;
    rayTracingLink Link;
    attachments finalAttachment;

    std::filesystem::path shadersPath;
    VkExtent2D extent;

    buffer stagingBuffer;
    VkCommandPool commandPool{VK_NULL_HANDLE};

public:
    rayTracingGraphics(const std::filesystem::path& shadersPath, VkExtent2D extent)
        : shadersPath(shadersPath), extent(extent)
    {
        setExtent(extent);
        Link.setShadersPath(shadersPath);
        link = &Link;
    }

    void setPositionInWindow(const vector<float,2>& offset, const vector<float,2>& size) override {
        this->offset = offset;
        this->size = size;
        Link.setPositionInWindow(offset, size);
    }

    ~rayTracingGraphics(){
        rayTracingGraphics::destroy();
    }

    void setExtent(VkExtent2D extent){
        this->extent = extent;
        rayTracer.setExtent(extent.width, extent.height);
    }
    void bind(const cuda::model* m) {
        rayTracer.bind(m);
    }
    void setCamera(cuda::camera* cam){
        rayTracer.setCamera(cam);
    }

    void create() override;
    void destroy() override;

    std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) override;

    void update(uint32_t) override {}

    void clearFrame(){
        rayTracer.clearFrame();
    }
};

#endif // !RAYTRACINGGRAPHICS

