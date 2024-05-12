#ifndef CUDARAYTRACING
#define CUDARAYTRACING

#include "math/vec4.h"
#include "transformational/camera.h"
#include "transformational/object.h"
#include "utils/buffer.h"
#include "utils/kdTree.h"

#include <stdint.h>

namespace cuda::rayTracing {

struct FrameRecord{
    HitRecord hit;
    vec4f color;
    vec4f bloom;
};

class RayTracing {
public:
    using Container_host = KDTree<std::vector<const Primitive*>>;
    using Container_dev = HitableKDTree;

private:
    uint32_t width;
    uint32_t height;
    Buffer<FrameRecord> record;
    Buffer<uint32_t> baseColor;
    Buffer<uint32_t> bloomColor;
    Buffer<curandState> randState;

    uint32_t xThreads{ 8 };
    uint32_t yThreads{ 8 };
    uint32_t minRayIterations{ 2 };
    uint32_t maxRayIterations{ 12 };

    bool clear{false};

    Devicep<Camera>* cam{nullptr};

    Devicep<Container_dev> devContainer;
    Container_host hostContainer;

public:

    RayTracing();
    ~RayTracing();

    void setExtent(uint32_t width, uint32_t height){
        this->width = width;
        this->height = height;
    }
    void bind(Object* obj) {
        obj->model->load(obj->transform);
        for(const auto& primitive : obj->model->primitives){
            hostContainer.storage.push_back(&primitive);
        }
    }
    void setCamera(Devicep<Camera>* cam){
        this->cam = cam;
    }

    void create();
    void update();

    bool calculateImage(uint32_t* baseColor, uint32_t* bloomColor);

    void clearFrame(){
        clear = true;
    }

    void buildTree();

    KDTree<std::vector<const Primitive*>>& getTree(){
        return hostContainer;
    }
};

}

#endif // !CUDARAYTRACING

