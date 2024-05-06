#ifndef TRIANGLE
#define TRIANGLE

#include "hitable.h"

namespace cuda::rayTracing {

struct Vertex {
    vec4f point{0.0f, 0.0f, 0.0f, 1.0f};
    vec4f normal{0.0f, 0.0f, 0.0f, 0.0f};
    vec4f color{0.0f, 0.0f, 0.0f, 0.0f};
    Properties props;
    __host__ __device__ Vertex() {}
    __host__ __device__ Vertex(const vec4f& point, const vec4f& normal, const vec4f& color, const Properties& props):
        point(point), normal(normal), color(color), props(props)
    {}
};

class Triangle : public Hitable {
private:
    size_t index[3];
    const Vertex* vertexBuffer{ nullptr };

public:
    __host__ __device__ Triangle() {}
    __host__ __device__ ~Triangle() {}

    __host__ __device__ Triangle(const size_t& i0, const size_t& i1, const size_t& i2, const Vertex* vertexBuffer);
    __host__ __device__ bool hit(const ray& r, HitCoords& coords) const override;
    __host__ __device__ void calcHitRecord(const ray& r, const HitCoords& coords, HitRecord& rec) const override;

    static void create(Triangle* dpointer, const Triangle& host);
    static void destroy(Triangle* dpointer);
    __host__ __device__ box getBox() const override;
};

}

#endif
