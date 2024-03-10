#ifndef HITABLEH
#define HITABLEH

#include "math/ray.h"
#include "materials/material.h"

namespace cuda {

    struct hitRecord{
        vec4 point{0.0f, 0.0f, 0.0f, 1.0f };
        vec4 normal{0.0f, 0.0f, 0.0f, 0.0f };
        vec4 color{1.0f, 1.0f, 1.0f, 1.0f };
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
        vec4 min{0.0f};
        vec4 max{0.0f};
    };

    class hitable {
    public:
        box bbox;

        __host__ __device__ virtual ~hitable() {};
        __host__ __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitCoords& coords) const = 0;
        __host__ __device__ virtual void calcHitRecord(const ray& r, const hitCoords& coords, hitRecord& rec) const = 0;
    };

}

#endif
