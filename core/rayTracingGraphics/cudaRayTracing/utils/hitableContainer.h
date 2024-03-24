#ifndef HITABLECONTAINER_H
#define HITABLECONTAINER_H

#include <vector>
#include "hitable.h"

namespace cuda {

    class hitableContainer {
    protected:
        size_t container_size{0};

    public:
        __host__ __device__ virtual ~hitableContainer(){}
        __host__ __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const = 0;

        __host__ __device__ virtual void add(hitable* objects) = 0;

        __host__ __device__ virtual hitable* operator[](uint32_t i) = 0;
        __host__ __device__ virtual size_t size() const {
            return container_size;
        }

        static void destroy(hitableContainer* dpointer);
    };

    void add(hitableContainer* container, const std::vector<hitable*>& objects);
}

#endif // HITABLECONTAINER_H
