#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class alignas(64) sphere : public hitable {
private:
    vec4 center{ 0.0f, 0.0f, 0.0f, 1.0f };
    vec4 color{ 0.0f,0.0f, 0.0f, 0.0f };
    float radius{ 0.0f };
    properties props;

public:
    __host__ __device__ sphere() {}
    __host__ __device__ ~sphere() {}

    __host__ __device__ sphere(vec4 cen, float r, vec4 color, const properties& props) : center(cen), radius(r), color(color), props(props) {};
    __host__ __device__ sphere(vec4 cen, float r, vec4 color) : center(cen), radius(r), color(color) {};
    __host__ __device__ bool hit(const ray& r, float tMin, float tMax, hitCoords& coords) const override;
    __host__ __device__ hitRecord calcHitRecord(const ray& r, const hitCoords& coords) const override;

    static sphere* create(vec4 cen, float r, vec4 color, const properties& props);
};

#endif
