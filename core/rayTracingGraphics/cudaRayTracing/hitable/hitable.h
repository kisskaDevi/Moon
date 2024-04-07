#ifndef HITABLEH
#define HITABLEH

#include "math/ray.h"
#include "materials/material.h"
#include "utils/devicep.h"

#include <stdint.h>
#include <algorithm>
#include <vector>

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

    struct box{
        alignas(16) vec4f min{std::numeric_limits<float>::max()};
        alignas(16) vec4f max{std::numeric_limits<float>::min()};
    };

    struct cbox : public box{
        alignas(16) vec4f color{0.0f, 1.0f, 0.0f, 0.0f};
        __host__ __device__ cbox(){}
        __host__ __device__ cbox(const box& b) : box(b){}
        __host__ __device__ cbox(const box& b, const vec4f& color) : box(b), color(color){}

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

    class hitable {
    public:
        __host__ __device__ virtual ~hitable() {};
        __host__ __device__ virtual bool hit(const ray& r, hitCoords& coords) const = 0;
        __host__ __device__ virtual void calcHitRecord(const ray& r, const hitCoords& coords, hitRecord& rec) const = 0;
        __host__ __device__ virtual box calcBox() const = 0;

        static void destroy(hitable* dpointer);
    };

    struct primitive{
        devicep<hitable> hit;
        cbox bbox;

        box calcBox() const{
            return bbox;
        }
    };

    template <typename iterator>
    __host__ __device__ box calcBox(iterator begin, iterator end){
        box resbox;
        for(auto it = begin; it != end; it++){
            resbox.min = cuda::min((*it)->calcBox().min, resbox.min);
            resbox.max = cuda::max((*it)->calcBox().max, resbox.max);
        }
        return resbox;
    }

    inline void sort(std::vector<const cuda::primitive*>::iterator begin, size_t size, cbox& box){
        std::vector<const cuda::primitive*>::iterator end = begin + size;
        box = calcBox(begin, end);

        vec4f limits = box.max - box.min;
        std::sort(begin, end, [i = limits.maxValueIndex(3)](const cuda::primitive* a, const cuda::primitive* b){
            return a->calcBox().min[i] < b->calcBox().min[i];
        });
    }
}

#endif
