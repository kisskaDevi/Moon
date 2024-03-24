#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

namespace cuda {

    class alignas(64) sphere : public hitable {
    private:
        vec4 center{ 0.0f, 0.0f, 0.0f, 1.0f };
        vec4 color{ 0.0f,0.0f, 0.0f, 0.0f };
        float radius{ 0.0f };
        properties props;

    public:
        __host__ __device__ sphere() {}
        __host__ __device__ ~sphere() {}

        __host__ __device__ sphere(const vec4& cen, float r, const vec4& color, const properties& props);
        __host__ __device__ sphere(const vec4& cen, float r, const vec4& color);
        __host__ __device__ bool hit(const ray& r, float tMin, float tMax, hitCoords& coords) const override;
        __host__ __device__ void calcHitRecord(const ray& r, const hitCoords& coords, hitRecord& rec) const override;

        static void create(sphere* dpointer, const sphere& host);
        static void destroy(sphere* dpointer);
        box calcBox() const;
    };

}

#endif
