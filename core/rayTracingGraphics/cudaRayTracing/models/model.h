#ifndef MODELH
#define MODELH

#include "utils/buffer.h"
#include "utils/primitive.h"
#include "hitable/triangle.h"

namespace cuda {

    struct model {
        buffer<vertex> vertexBuffer;
        std::vector<primitive> primitives;

        model(const std::vector<vertex>& vertexBuffer, const std::vector<uint32_t>& indexBuffer);
        model(std::vector<primitive>&& primitives);
        model(primitive&& primitive);
        model() = default;
        model(model&& m);
        model& operator=(model&& m);

        ~model();
    };
}

#endif
