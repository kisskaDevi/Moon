#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitableContainer.h"

namespace cuda {

    class hitableList : public hitableContainer {
    public:
        struct node{
            hitable* current{nullptr};
            node* next{nullptr};
            __host__ __device__ node* get_next() { return next; }
            __host__ __device__ hitable*& operator()() { return current; }
        };

    private:

        node* head{nullptr};
        node* tail{nullptr};

    public:
        using iterator = baseIterator<node>;

        __host__ __device__ hitableList(){};
        __host__ __device__ ~hitableList();

        __host__ __device__ bool hit(const ray& r, hitCoords& coord) const override;

        __host__ __device__ void add(hitable* object) override ;

        __host__ __device__ hitable*& operator[](uint32_t i) const override;

        __host__ __device__ iterator begin() {return iterator(head); }
        __host__ __device__ iterator end() {return iterator(); }

        __host__ __device__ iterator begin() const {return iterator(head); }
        __host__ __device__ iterator end() const {return iterator(); }

        static void create(hitableList* dpointer, const hitableList& host);
        static void destroy(hitableList* dpointer);
    };

    __host__ __device__ inline void sort(hitableList::iterator begin, size_t size, cbox& box){
        hitableList::iterator last = begin + size;
        for(auto it = begin; it != last; it++){
            box.min = cuda::min((*it)->calcBox().min, box.min);
            box.max = cuda::max((*it)->calcBox().max, box.max);
        }
    }
}

#endif
