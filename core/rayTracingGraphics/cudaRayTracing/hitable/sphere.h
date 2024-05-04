#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

namespace cuda {

    class sphere : public hitable {
    private:
        vec4f center{ 0.0f, 0.0f, 0.0f, 1.0f };
        vec4f color{ 0.0f,0.0f, 0.0f, 0.0f };
        float radius{ 0.0f };
        properties props;

    public:
        __host__ __device__ sphere() {}
        __host__ __device__ ~sphere() {}

        __host__ __device__ sphere(const vec4f& cen, float r, const vec4f& color, const properties& props);
        __host__ __device__ sphere(const vec4f& cen, float r, const vec4f& color);
        __host__ __device__ bool hit(const ray& r, hitCoords& coords) const override;
        __host__ __device__ void calcHitRecord(const ray& r, const hitCoords& coords, hitRecord& rec) const override;

        static void create(sphere* dpointer, const sphere& host);
        static void destroy(sphere* dpointer);
        __host__ __device__ box getBox() const override;
    };

}

#endif
