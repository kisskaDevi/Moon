#ifndef HITABLECONTAINER_H
#define HITABLECONTAINER_H

#include <vector>
#include "hitable.h"

class hitableContainer {
public:
    __host__ __device__ virtual ~hitableContainer(){}
    __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const = 0;

    __host__ __device__ virtual void add(hitable* objects) = 0;
};

void add(hitableContainer* container, const std::vector<hitable*>& objects);

#endif // HITABLECONTAINER_H
