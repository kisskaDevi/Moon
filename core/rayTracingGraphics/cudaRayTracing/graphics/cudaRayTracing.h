#ifndef CUDARAYTRACING
#define CUDARAYTRACING

#include "math/vec4.h"
#include "transformational/camera.h"
#include "utils/hitableContainer.h"
#include "utils/buffer.h"

#include <stdint.h>

struct fragment{
    vec4 color;
    hitRecord record;
};

struct frameBuffer{
    fragment base;
    fragment bloom;
};

class cudaRayTracing {
private:
    cuda::buffer<frameBuffer> frame;
    cuda::buffer<uint32_t> swapChainImage;

    uint32_t xThreads{ 8 };
    uint32_t yThreads{ 8 };
    uint32_t minRayIterations{ 2 };
    uint32_t maxRayIterations{ 12 };

    bool clear{false};

    cuda::camera* cam{nullptr};
    hitableContainer* container{nullptr};
    uint32_t* hostFrameBuffer{nullptr};

    uint32_t width;
    uint32_t height;

public:
    cudaRayTracing(){}

    ~cudaRayTracing(){
        destroy();
    }

    void setExtent(uint32_t width, uint32_t height){
        this->width = width;
        this->height = height;
    }
    void setList(hitableContainer* container) {
        this->container = container;
    }
    void setCamera(cuda::camera* cam){
        this->cam = cam;
    }

    void create();
    void destroy();

    void calculateImage(uint32_t* hostFrameBuffer);

    void clearFrame(){
        clear = true;
    }
};

#endif // !CUDARAYTRACING

