#ifndef CUDARAYTRACING
#define CUDARAYTRACING

#include "math/vec4.h"
#include "transformational/camera.h"
#include "utils/hitableContainer.h"
#include "utils/buffer.h"
#include "models/model.h"

#include <stdint.h>

namespace cuda {

    struct fragment{
        vec4 color;
        cuda::hitRecord record;
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

        cuda::devicep<cuda::camera>* cam{nullptr};
        cuda::devicep<cuda::hitableContainer> container;
        uint32_t* hostFrameBuffer{nullptr};

        uint32_t width;
        uint32_t height;

    public:
        cudaRayTracing();
        ~cudaRayTracing();

        void setExtent(uint32_t width, uint32_t height){
            this->width = width;
            this->height = height;
        }
        void bind(cuda::model* m) {
            std::vector<hitable*> hitables;
            for(const auto& primitive : m->primitives){
                hitables.push_back(primitive.hit());
            }
            add(container.get(), hitables);
        }
        void setCamera(cuda::devicep<cuda::camera>* cam){
            this->cam = cam;
        }

        void create();
        void update();

        bool calculateImage(uint32_t* hostFrameBuffer);

        void clearFrame(){
            clear = true;
        }
    };

}

#endif // !CUDARAYTRACING

