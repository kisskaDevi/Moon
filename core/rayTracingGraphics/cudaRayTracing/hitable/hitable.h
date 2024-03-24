#ifndef HITABLEH
#define HITABLEH

#include "math/ray.h"
#include "materials/material.h"
#include <stdint.h>

namespace cuda {

    struct hitRecord{
        vec4f point{0.0f, 0.0f, 0.0f, 1.0f };
        vec4f normal{0.0f, 0.0f, 0.0f, 0.0f };
        vec4f color{1.0f, 1.0f, 1.0f, 1.0f };
        properties props;
        ray r;
        uint32_t rayDepth{0};
        float lightIntensity{1.0};
    };

    struct hitCoords{
        float t{0};
        float u{0};
        float v{0};
    };

    struct box{
        alignas(16) vec4f min{0.0f};
        alignas(16) vec4f max{0.0f};
    };

    struct cbox : public box{
        alignas(16) vec4f color{0.0f, 1.0f, 0.0f, 0.0f};
        cbox(const box& b) : box(b){}
        cbox(const box& b, const vec4f& color) : box(b), color(color){}
    };

    class hitable {
    public:
        __host__ __device__ virtual ~hitable() {};
        __host__ __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitCoords& coords) const = 0;
        __host__ __device__ virtual void calcHitRecord(const ray& r, const hitCoords& coords, hitRecord& rec) const = 0;

        static void destroy(hitable* dpointer);
    };

}

#endif
