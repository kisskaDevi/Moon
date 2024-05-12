#ifndef MODELH
#define MODELH

#include "utils/buffer.h"
#include "utils/primitive.h"
#include "hitable/triangle.h"
#include "math/mat4.h"

namespace cuda::rayTracing {

class Model {
public:
    Buffer<Vertex> vertexBuffer;
    std::vector<Primitive> primitives;

    Model(const std::vector<Vertex>& vertexBuffer, const std::vector<uint32_t>& indexBuffer);
    Model(std::vector<Primitive>&& primitives);
    Model(Primitive&& primitive);
    Model() = default;
    Model(Model&& m);
    Model& operator=(Model&& m);

    ~Model();

    virtual void load(const mat4f& transform) {}
};

}
#endif
