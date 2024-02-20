#ifndef HITABLEH
#define HITABLEH

#include "ray.h"
#include "material.h"

struct hitRecord
{
    vec4 point{0.0f, 0.0f, 0.0f, 1.0f };
    vec4 normal{0.0f, 0.0f, 0.0f, 0.0f };
    vec4 color{0.0f, 0.0f, 0.0f, 0.0f };
    properties props;
};

struct hitCoords
{
    float t{0};
    float u{0};
    float v{0};
    float s{0};
};

class hitable {
public:
    hitable* next{ nullptr };
    __host__ __device__ virtual ~hitable() {};
    __host__ __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitCoords& coords) const = 0;
    __host__ __device__ virtual void calcHitRecord(const ray& r, const hitCoords& coords, hitRecord& rec) const = 0;
};

#endif
