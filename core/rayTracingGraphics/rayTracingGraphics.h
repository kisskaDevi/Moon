#ifndef RAYTRACINGGRAPHICS
#define RAYTRACINGGRAPHICS

#include "rayTracingLink.h"
#include "graphicsInterface.h"
#include "attachments.h"
#include "buffer.h"
#include "texture.h"
#include "vector.h"

#include <stdint.h>

#include "cudaRayTracing.h"
#include "boundingBoxGraphics.h"

namespace cuda {
class model;
}

class rayTracingGraphics : public graphicsInterface {
private:
    uint32_t* hostFrameBuffer{nullptr};

    cuda::cudaRayTracing rayTracer;
    boundingBoxGraphics bbGraphics;
    rayTracingLink Link;

    attachments finalAttachment;
    texture* emptyTexture{nullptr};

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
        bbGraphics.destroy();
    }

    void setEnableBoundingBox(bool enable);
    void setExtent(VkExtent2D extent){
        this->extent = extent;
        rayTracer.setExtent(extent.width, extent.height);
    }
    void bind(cuda::model* m) {
        rayTracer.bind(m);
        bbGraphics.bind(m);
    }
    void setCamera(cuda::devicep<cuda::camera>* cam){
        rayTracer.setCamera(cam);
        bbGraphics.bind(cam);
    }

    void create() override;
    void destroy() override;
    void update(uint32_t imageIndex) override;
    std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) override;

    void clearFrame(){
        rayTracer.clearFrame();
    }
};

#endif // !RAYTRACINGGRAPHICS

