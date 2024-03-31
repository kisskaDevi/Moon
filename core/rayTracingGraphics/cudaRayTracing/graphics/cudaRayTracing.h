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

    struct fragment{
        vec4f color;
        cuda::hitRecord record;
    };

    struct frameBuffer{
        fragment base;
        fragment bloom;
    };

    class cudaRayTracing {
    public:
        using container_host = std::vector<const cuda::primitive*>;
        using container_dev = hitableList;
        using kdTree_host = kdNode<container_host::iterator>;
        using kdTree_dev = kdNode<container_dev::iterator>;

    private:
        cuda::buffer<frameBuffer> frame;
        cuda::buffer<uint32_t> swapChainImage;

        uint32_t xThreads{ 8 };
        uint32_t yThreads{ 8 };
        uint32_t minRayIterations{ 2 };
        uint32_t maxRayIterations{ 12 };

        bool clear{false};

        devicep<cuda::camera>* cam{nullptr};

        devicep<container_dev> devContainer;
        container_host hostContainer;

        devicep<kdTree_dev> devTree;
        kdTree_host* hostTree{nullptr};

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
            for(const auto& primitive : m->primitives){
                hostContainer.push_back(&primitive);
            }
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

        void buildTree();

        kdTree_host* getTree(){
            return hostTree;
        }
    };

}

#endif // !CUDARAYTRACING

