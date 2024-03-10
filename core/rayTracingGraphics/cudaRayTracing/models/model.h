#ifndef MODELH
#define MODELH

#include <vector>

#include "utils/buffer.h"
#include "hitable/triangle.h"

namespace cuda {

    class model {
    public:
        cuda::buffer<vertex> vertexBuffer;
        std::vector<hitable*> hitables;
        std::vector<box> boxes;

        model(const std::vector<vertex>& vertexBuffer, const std::vector<uint32_t>& indexBuffer);
        model(const std::vector<hitable*>& hitables, const std::vector<box>& boxes);
        model() = default;
        model(model&& m) = default;
        model& operator=(model&& m) = default;

        ~model();
    };
}

#endif
