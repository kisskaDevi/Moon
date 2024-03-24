#ifndef TRIANGLE
#define TRIANGLE

#include "hitable.h"

namespace cuda {

    struct vertex {
        vec4f point{0.0f, 0.0f, 0.0f, 1.0f};
        vec4f normal{0.0f, 0.0f, 0.0f, 0.0f};
        vec4f color{0.0f, 0.0f, 0.0f, 0.0f};
        properties props;
        __host__ __device__ vertex() {}
        __host__ __device__ vertex(vec4f point, vec4f normal, vec4f color, const properties& props):
            point(point), normal(normal), color(color), props(props)
        {}
    };

    class triangle : public hitable {
    private:
        size_t index[3];
        const vertex* vertexBuffer{ nullptr };

    public:
        __host__ __device__ triangle() {}
        __host__ __device__ ~triangle() {}

        __host__ __device__ triangle(const size_t& i0, const size_t& i1, const size_t& i2, const vertex* vertexBuffer);
        __host__ __device__ bool hit(const ray& r, float tMin, float tMax, hitCoords& rec) const override;
        __host__ __device__ void calcHitRecord(const ray& r, const hitCoords& coords, hitRecord& rec) const override;


        static void create(triangle* dpointer, const triangle& host);
        static void destroy(triangle* dpointer);
        box calcBox() const;
    };

}

#endif
