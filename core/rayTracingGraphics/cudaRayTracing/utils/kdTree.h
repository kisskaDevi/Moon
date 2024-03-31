#ifndef KDTREE_H
#define KDTREE_H

#include "hitable/hitable.h"

namespace cuda{

template <typename iterator>
struct kdNode{
    kdNode* left{nullptr};
    kdNode* right{nullptr};
    iterator begin;
    size_t size{0};
    cbox box;

    __host__ __device__ kdNode(iterator begin, size_t size) : begin(begin), size(size){
        sort(begin, size, box);

        if(size > 2){
            left = new kdNode(begin, size / 2);
            right = new kdNode(begin + left->size, size - left->size);
        }
    }

    __host__ __device__ ~kdNode(){
        if(left){
            delete left;
        }
        if(right){
            delete right;
        }
    }

    __host__ __device__ bool hit(const ray& r, hitCoords& coord) const {
        if(!box.intersect(r)){
            return false;
        }

        if(!(left || right)){
            for(iterator it = begin; it != (begin + size); it++){
                if ((*it)->hit(r, coord)) {
                    coord.obj = *it;
                }
            }
            return coord.obj;
        }

        bool res = false;
        res |= left && left->hit(r, coord);
        res |= right && right->hit(r, coord);
        return res;
    }
};

}

#endif // KDTREE_H
