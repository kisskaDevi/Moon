#ifndef CUDARAYTRACING
#define CUDARAYTRACING

#include "math/vec4.h"
#include "transformational/camera.h"
#include "utils/buffer.h"
#include "utils/kdTree.h"
#include "models/model.h"

#include <stdint.h>

#include "hitableArray.h"
#include "hitableList.h"

namespace cuda {
    struct frameRecord{
        cuda::hitRecord hit;
        vec4f color;
        vec4f bloom;
    };

    class cudaRayTracing {
    public:
        using container_host = kdTree<std::vector<const cuda::primitive*>>;
        using container_dev = hitableKDTree;

    private:
        uint32_t width;
        uint32_t height;
        cuda::buffer<frameRecord> record;
        cuda::buffer<uint32_t> baseColor;
        cuda::buffer<uint32_t> bloomColor;

        uint32_t xThreads{ 8 };
        uint32_t yThreads{ 8 };
        uint32_t minRayIterations{ 2 };
        uint32_t maxRayIterations{ 12 };

        bool clear{false};

        devicep<cuda::camera>* cam{nullptr};

        devicep<container_dev> devContainer;
        container_host hostContainer;

    public:

        cudaRayTracing();
        ~cudaRayTracing();

        void setExtent(uint32_t width, uint32_t height){
            this->width = width;
            this->height = height;
        }
        void bind(cuda::model* m) {
            for(const auto& primitive : m->primitives){
                hostContainer.storage.push_back(&primitive);
            }
        }
        void setCamera(cuda::devicep<cuda::camera>* cam){
            this->cam = cam;
        }

        void create();
        void update();

        bool calculateImage(uint32_t* baseColor, uint32_t* bloomColor);

        void clearFrame(){
            clear = true;
        }

        void buildTree();

        kdTree<std::vector<const cuda::primitive*>>& getTree(){
            return hostContainer;
        }
    };

}

#endif // !CUDARAYTRACING

