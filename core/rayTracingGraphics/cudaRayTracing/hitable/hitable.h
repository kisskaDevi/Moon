#ifndef HITABLEH
#define HITABLEH

#include "math/box.h"
#include "materials/material.h"

namespace cuda {

    struct hitRecord{
        vec4f point{0.0f, 0.0f, 0.0f, 1.0f };
        vec4f normal{0.0f, 0.0f, 0.0f, 0.0f };
        vec4f color{1.0f, 1.0f, 1.0f, 1.0f };
        properties props;
        ray scattering;
        uint32_t rayDepth{0};
        float lightIntensity{1.0};
    };

    class hitable;

    struct hitCoords{
        float tmin{0.01f};
        float tmax{std::numeric_limits<float>::max()};
        float u{0.0f};
        float v{0.0f};
        hitable* obj{nullptr};

        __host__ __device__ bool check() const {
            return obj && tmax != std::numeric_limits<float>::max();
        }
    };

    class hitable {
    public:
        __host__ __device__ virtual ~hitable() {};
        __host__ __device__ virtual bool hit(const ray& r, hitCoords& coords) const = 0;
        __host__ __device__ virtual void calcHitRecord(const ray& r, const hitCoords& coords, hitRecord& rec) const = 0;
        __host__ __device__ virtual box getBox() const = 0;

        static void destroy(hitable* dpointer);
    };
}

#endif
