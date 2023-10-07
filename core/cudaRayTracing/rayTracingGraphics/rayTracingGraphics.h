#ifndef RAYTRACINGGRAPHICS
#define RAYTRACINGGRAPHICS

#include "graphicsInterface.h"
#include "core/utils/device.h"
#include "core/utils/swapChain.h"
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
    uint32_t rayDepth{ 8 };

    bool clear{false};

    cuda::camera* cam{nullptr};
    curandState* randState{nullptr};
    hitableContainer* container{nullptr};
    uint32_t* hostFrameBuffer{nullptr};

    std::vector<physicalDevice*> devices;
    physicalDevice* device{nullptr};
    swapChain* swapChainKHR{nullptr};

    rayTracingLink Link;
    attachments finalAttachment;

    std::filesystem::path shadersPath;
    VkExtent2D extent;
    VkOffset2D offset;

    buffer stagingBuffer;
    VkCommandPool commandPool{VK_NULL_HANDLE};

public:
    rayTracingGraphics(const std::filesystem::path& shadersPath, VkExtent2D extent, VkOffset2D offset)
        : shadersPath(shadersPath), extent(extent), offset(offset), cam(cam){
        Link.setShadersPath(shadersPath);
    }

    void setDevices(uint32_t devicesCount, physicalDevice* devices) override {
        this->devices.resize(devicesCount);
        for(uint32_t i = 0 ; i < devicesCount; i++){
            this->devices[i] = &devices[i];
        }
        device = this->devices.front();
        Link.setDeviceProp(device->getLogical());
    }
    void setSwapChain(swapChain* swapChainKHR) override {
        this->swapChainKHR = swapChainKHR;
        Link.setImageCount(swapChainKHR->getImageCount());
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

    void createGraphics() override;
    void destroyGraphics() override;

    std::vector<std::vector<VkSemaphore>> submit(const std::vector<std::vector<VkSemaphore>>& externalSemaphore, const std::vector<VkFence>& externalFence, uint32_t imageIndex) override;

    inline uint32_t* getSwapChain() {
        return swapChainImage.get();
	}
	size_t getWidth() const {
        return swapChainKHR->getExtent().width;
	}
	size_t getHeight() const {
        return swapChainKHR->getExtent().height;
	}
    linkable* getLinkable() override {
        return &Link;
    }

    void updateCommandBuffer(uint32_t) override {}
    void updateBuffers(uint32_t) override {}
};

#endif // !RAYTRACINGGRAPHICS

