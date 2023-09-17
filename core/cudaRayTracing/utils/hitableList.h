#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitableContainer.h"

class hitableList : public hitableContainer {
private:
    hitable* head{ nullptr };
    hitable* tail{ nullptr };

public:
    __host__ __device__ hitableList() {}
    __host__ __device__ ~hitableList();

    __device__ bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const override;

    __host__ __device__ void add(hitable* object) override ;

    static hitableList* create();
    static void destroy(hitableList* list);
};

#endif
