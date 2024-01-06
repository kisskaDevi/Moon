#ifndef RAYTRACINGGRAPHICS
#define RAYTRACINGGRAPHICS

#include "graphicsInterface.h"
#include "core/utils/attachments.h"
#include "core/utils/buffer.h"

#include "rayTracingLink.h"
#include "vec4.h"
#include "camera.h"
#include "hitableContainer.h"
#include "buffer.h"

#include <stdint.h>

class rayTracingGraphics : public graphicsInterface {
private:
    cuda::buffer<vec4> bloomImage;
    cuda::buffer<vec4> colorImage;
    cuda::buffer<uint32_t> swapChainImage;

    uint32_t width{0};
    uint32_t height{0};
    uint32_t xThreads{ 8 };
    uint32_t yThreads{ 8 };
    uint32_t rayDepth{ 12 };

    bool clear{false};

    cuda::camera* cam{nullptr};
    curandState* randState{nullptr};
    hitableContainer* container{nullptr};
    uint32_t* hostFrameBuffer{nullptr};

    rayTracingLink Link;
    attachments finalAttachment;

    std::filesystem::path shadersPath;
    VkExtent2D extent;
    VkOffset2D offset;

    buffer stagingBuffer;
    VkCommandPool commandPool{VK_NULL_HANDLE};

public:
    rayTracingGraphics(const std::filesystem::path& shadersPath, VkExtent2D extent, VkOffset2D offset)
        : shadersPath(shadersPath), extent(extent), offset(offset), cam(cam)
    {
        Link.setShadersPath(shadersPath);
        link = &Link;
    }

    ~rayTracingGraphics(){
        rayTracingGraphics::destroy();
    }

    void setList(hitableContainer* container) {
        this->container = container;
    }
    void setCamera(cuda::camera* cam){
        this->cam = cam;
    }
    void update(){
        clear = true;
    }

    void create() override;
    void destroy() override;

    std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) override;

    inline uint32_t* getSwapChain() {
        return swapChainImage.get();
	}
	size_t getWidth() const {
        return swapChainKHR->getExtent().width;
	}
	size_t getHeight() const {
        return swapChainKHR->getExtent().height;
    }

    void update(uint32_t) override {}
};

#endif // !RAYTRACINGGRAPHICS

