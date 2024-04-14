#ifndef KDTREE_H
#define KDTREE_H

#include "hitableArray.h"
#include "utils/stack.h"

namespace cuda{

template <typename iterator>
struct kdNode{
    kdNode* left{nullptr};
    kdNode* right{nullptr};
    iterator begin;
    size_t size{0};
    cbox box;

    __host__ __device__ void del(){
        if(left){
            delete left;
        }
        if(right){
            delete right;
        }
    }

    __host__ __device__ kdNode(){}
    __host__ __device__ kdNode(const kdNode& other) = delete;
    __host__ __device__ kdNode& operator=(const kdNode& other) = delete;

    __host__ __device__ kdNode(kdNode&& other) = default;
    __host__ __device__ kdNode& operator=(kdNode&& other)
    {
        left = other.left;
        other.left = nullptr;
        right = other.right;
        other.right = nullptr;
        begin = other.begin;
        size = other.size;
        box = other.box;
        return *this;
    }

    __host__ __device__ kdNode(iterator begin, size_t size) : begin(begin), size(size){
        sort(begin, size, box);

        if(size > 10){
            left = new kdNode(begin, size / 2);
            right = new kdNode(begin + left->size, size - left->size);
        }
    }

    __host__ __device__ ~kdNode() { del(); }

    __host__ __device__ bool hit(const ray& r, hitCoords& coord) const {
        stack<const kdNode*, 50> selected;
        for(stack<const kdNode*, 50> treeTraverse(this); !treeTraverse.empty();){
            const kdNode* curr = treeTraverse.top();
            treeTraverse.pop();

            if(curr->box.intersect(r)){
                if(curr->left){
                    treeTraverse.push(curr->left);
                }
                if(curr->right){
                    treeTraverse.push(curr->right);
                }
                if(!(curr->left || curr->right)){
                    selected.push(curr);
                }
            }
        }
        for(;!selected.empty();){
            const kdNode* curr = selected.top();
            selected.pop();
            for(iterator it = curr->begin; it != (curr->begin + curr->size); it++){
                if ((*it)->hit(r, coord)) {
                    coord.obj = *it;
                }
            }
        }

        return coord.obj;
    }
};

class kdTree : public hitableContainer {
public:
    using container = hitableArray;

private:
    container* storage{nullptr};
    kdNode<container::iterator>* root{nullptr};

public:
    using iterator = container::iterator;

    __host__ __device__ kdTree() {
        storage = new container();
    };
    __host__ __device__ ~kdTree(){
        if(storage){
            delete storage;
        }
        if(root){
            delete root;
        }
    }

    __host__ __device__ bool hit(const ray& r, hitCoords& coord) const override{
        return root->hit(r, coord);
    }

    __host__ __device__ void add(hitable* object) override{
        storage->add(object);
    }

    __host__ __device__ hitable*& operator[](uint32_t i) const override{
        return (*storage)[i];
    }

    __host__ __device__ void makeTree() {
        root = new kdNode<container::iterator>(storage->begin(), storage->size());
    }

    __host__ __device__ iterator begin() {return storage->begin(); }
    __host__ __device__ iterator end() {return storage->end(); }

    __host__ __device__ iterator begin() const {return storage->begin(); }
    __host__ __device__ iterator end() const {return storage->end(); }

    static void create(kdTree* dpointer, const kdTree& host);
    static void destroy(kdTree* dpointer);
};

void makeTree(kdTree* container);
}

#endif // KDTREE_H
