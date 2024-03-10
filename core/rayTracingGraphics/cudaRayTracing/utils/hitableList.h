#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitableContainer.h"

namespace cuda {

    class hitableList : public hitableContainer {
    private:
        struct node{
            hitable* current{nullptr};
            node* next{nullptr};
        };

        node* head{nullptr};
        node* tail{nullptr};

    public:
        __host__ __device__ hitableList(){};
        __host__ __device__ ~hitableList();

        __host__ __device__ bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const override;

        __host__ __device__ void add(hitable* object) override ;

        __host__ __device__ hitable* operator[](uint32_t i) override;

        static hitableList* create();
    };
}

#endif
