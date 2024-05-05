#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitableContainer.h"

namespace cuda::rayTracing {

class HitableList : public HitableContainer {
public:
    struct Node{
        Hitable* current{nullptr};
        Node* next{nullptr};
        __host__ __device__ Node* get_next() { return next; }
        __host__ __device__ Hitable*& operator()() { return current; }
    };

private:
    Node* head{nullptr};
    Node* tail{nullptr};

public:
    using iterator = BaseIterator<Node>;

    __host__ __device__ HitableList(){};
    virtual __host__ __device__ ~HitableList();

    __host__ __device__ bool hit(const ray& r, HitCoords& coord) const override;

    __host__ __device__ void add(Hitable* object) override ;

    __host__ __device__ Hitable*& operator[](uint32_t i) const override;

    __host__ __device__ iterator begin() {return iterator(head); }
    __host__ __device__ iterator end() {return iterator(); }

    __host__ __device__ iterator begin() const {return iterator(head); }
    __host__ __device__ iterator end() const {return iterator(); }

    static void create(HitableList* dpointer, const HitableList& host);
    static void destroy(HitableList* dpointer);
};

}

#endif
