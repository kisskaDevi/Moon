#ifndef HITABLECONTAINER_H
#define HITABLECONTAINER_H

#include <vector>
#include "hitable.h"

namespace cuda {

    class hitableContainer {
    protected:
        size_t container_size{0};

    public:
        template <typename type>
        class baseIterator{
        protected:
            type* ptr{nullptr};

        public:
            typedef type value_type;

            __host__ __device__ baseIterator() {};
            __host__ __device__ baseIterator(type* ptr) : ptr(ptr) {}

            __host__ __device__ hitable*& operator*() const { return (*ptr)(); }
            __host__ __device__ hitable** operator->() { return ptr(); }
            __host__ __device__ baseIterator& operator++() { ptr = ptr->get_next(); return *this; }
            __host__ __device__ baseIterator operator++(int) { baseIterator tmp = *this; ++(*this); return tmp; }
            __host__ __device__ friend bool operator== (const baseIterator& a, const baseIterator& b) { return a.ptr == b.ptr; };
            __host__ __device__ friend bool operator!= (const baseIterator& a, const baseIterator& b) { return a.ptr != b.ptr; };
            __host__ __device__ friend baseIterator operator+ (baseIterator it, size_t s) {
                for(; s > 0; s--) it.ptr = it.ptr->get_next();
                return it;
            };
        };

        __host__ __device__ virtual ~hitableContainer(){}
        __host__ __device__ virtual bool hit(const ray& r, hitCoords& coord) const = 0;

        __host__ __device__ virtual void add(hitable* objects) = 0;

        __host__ __device__ virtual hitable*& operator[](uint32_t i) const = 0;
        __host__ __device__ virtual size_t size() const { return container_size; }

        static void destroy(hitableContainer* dpointer);
    };

    void add(hitableContainer* container, const std::vector<hitable*>& objects);
}

#endif // HITABLECONTAINER_H
