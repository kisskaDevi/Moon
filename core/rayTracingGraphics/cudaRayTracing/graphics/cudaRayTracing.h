#ifndef CUDARAYTRACING
#define CUDARAYTRACING

#include "math/vec4.h"
#include "transformational/camera.h"
#include "utils/hitableContainer.h"
#include "utils/buffer.h"

#include <stdint.h>

class cudaRayTracing {
private:
    cuda::buffer<vec4> bloomImage;
    cuda::buffer<vec4> colorImage;
    cuda::buffer<uint32_t> swapChainImage;

    uint32_t xThreads{ 8 };
    uint32_t yThreads{ 8 };
    uint32_t rayDepth{ 8 };

    bool clear{false};

    cuda::camera* cam{nullptr};
    curandState* randState{nullptr};
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

