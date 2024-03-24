#ifndef MODELH
#define MODELH

#include <vector>

#include "utils/buffer.h"
#include "hitable/triangle.h"
#include "utils/devicep.h"

namespace cuda {

    struct primitive{
        devicep<hitable> hit;
        cbox box;
    };

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

        void setBoxesColor(const vec4& color);
    };
}

#endif
