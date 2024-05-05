#ifndef MODELH
#define MODELH

#include "utils/buffer.h"
#include "utils/primitive.h"
#include "hitable/triangle.h"

namespace cuda::rayTracing {

struct Model {
    Buffer<Vertex> vertexBuffer;
    std::vector<Primitive> primitives;

    Model(const std::vector<Vertex>& vertexBuffer, const std::vector<uint32_t>& indexBuffer);
    Model(std::vector<Primitive>&& primitives);
    Model(Primitive&& primitive);
    Model() = default;
    Model(Model&& m);
    Model& operator=(Model&& m);

    ~Model();
};

}
#endif
