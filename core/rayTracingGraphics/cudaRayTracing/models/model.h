#ifndef MODELH
#define MODELH

#include <vector>

#include "utils/buffer.h"
#include "hitable/triangle.h"

namespace cuda {

    class model {
    public:
        buffer<vertex> vertexBuffer;
        std::vector<hitable*> hitables;
        std::vector<cbox> boxes;

        model(const std::vector<vertex>& vertexBuffer, const std::vector<uint32_t>& indexBuffer);
        model(const std::vector<hitable*>& hitables, const std::vector<cbox>& boxes  = {});
        model() = default;
        model(model&& m);
        model& operator=(model&& m);

        ~model();

        void setBoxesColor(const vec4& color);
    };
}

#endif
