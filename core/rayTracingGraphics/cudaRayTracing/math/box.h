#ifndef BOX_H
#define BOX_H
#include "ray.h"

namespace cuda::rayTracing {

struct box{
    vec4f min{std::numeric_limits<float>::max()};
    vec4f max{std::numeric_limits<float>::lowest()};

    __host__ __device__ float surfaceArea() const {
        const float dx = max.x() - min.x();
        const float dy = max.y() - min.y();
        const float dz = max.z() - min.z();
        return 2.0f * (dx * dy + dz * dy + dx * dz);
    }

    __host__ __device__ bool intersect(const ray &r) const {
        float dx = 1.0f / r.getDirection().x();
        float dy = 1.0f / r.getDirection().y();
        float dz = 1.0f / r.getDirection().z();

        float t1 = (min.x() - r.getOrigin().x()) * dx;
        float t2 = (max.x() - r.getOrigin().x()) * dx;
        float t3 = (min.y() - r.getOrigin().y()) * dy;
        float t4 = (max.y() - r.getOrigin().y()) * dy;
        float t5 = (min.z() - r.getOrigin().z()) * dz;
        float t6 = (max.z() - r.getOrigin().z()) * dz;

        float tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
        float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

        // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
        if (tmax < 0)
        {
            return false;
        }
        // if tmin > tmax, ray doesn't intersect AABB
        if (tmin > tmax)
        {
            return false;
        }
        return true;
    }
};

struct cbox : public box{
    vec4f color{0.0f, 0.0f, 0.0f, 0.0f};
    __host__ __device__ cbox() {}
    __host__ __device__ cbox(const box& b) : box(b) {}
    __host__ __device__ cbox(const box& b, const vec4f& color) : box(b), color(color) {}
};

}

#endif // BOX_H
