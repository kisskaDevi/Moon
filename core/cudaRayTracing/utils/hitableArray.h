#ifndef HITABLEARRAY_H
#define HITABLEARRAY_H

#include "hitableContainer.h"

class hitableArray : public hitableContainer {
private:
    hitable** array{ nullptr };
    size_t size{0};

public:
    __host__ __device__ hitableArray() {}
    __host__ __device__ ~hitableArray();

    __device__ bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const override;

    __host__ __device__ void add(hitable* object) override;

    static hitableArray* create();
    static void destroy(hitableArray* array);
};

#endif // HITABLEARRAY_H
