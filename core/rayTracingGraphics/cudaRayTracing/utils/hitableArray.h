#ifndef HITABLEARRAY_H
#define HITABLEARRAY_H

#include "hitableContainer.h"

namespace cuda {

    class hitableArray : public hitableContainer {
    private:
        hitable** array{ nullptr };

    public:
        __host__ __device__ hitableArray(){};
        __host__ __device__ ~hitableArray();

        __host__ __device__ bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const override;

        __host__ __device__ void add(hitable* object) override;

        __host__ __device__ hitable* operator[](uint32_t i) override;

        static hitableArray* create();
    };
}

#endif // HITABLEARRAY_H
