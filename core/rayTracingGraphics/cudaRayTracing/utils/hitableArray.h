#ifndef HITABLEARRAY_H
#define HITABLEARRAY_H

#include "hitableContainer.h"

namespace cuda {

    class hitableArray : public hitableContainer {
    public:
        struct pointer{
            hitable* p{nullptr};
            __host__ __device__ pointer* get_next() { return this + 1; }
            __host__ __device__ hitable*& operator()() { return p; }
        };

        using iterator = baseIterator<pointer>;

        __host__ __device__ hitableArray(){};
        __host__ __device__ ~hitableArray();

        __host__ __device__ bool hit(const ray& r, hitCoords& coord) const override;

        __host__ __device__ void add(hitable* object) override;

        __host__ __device__ hitable*& operator[](uint32_t i) const override;

        __host__ __device__ iterator begin() {return iterator(&array[0]); }
        __host__ __device__ iterator end() {return iterator(&array[container_size]); }

        __host__ __device__ iterator begin() const {return iterator(&array[0]); }
        __host__ __device__ iterator end() const {return iterator(&array[container_size]); }

        static void create(hitableArray* dpointer, const hitableArray& host);
        static void destroy(hitableArray* dpointer);

    private:
        pointer* array{ nullptr };
    };
}

#endif // HITABLEARRAY_H
