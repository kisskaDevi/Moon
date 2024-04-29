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

        if(size > 10)
        {
            float bestSAH = std::numeric_limits<float>::max();
            size_t bestSize = 0, itSize = 0;
            const iterator end = begin + size;
            for(auto curr = begin; curr != end; curr++){
                size_t leftN = ++itSize;
                size_t rightN = size - itSize;

                float leftS = calcBox(begin, curr + 1).surfaceArea();
                float rightS = calcBox(curr + 1, end).surfaceArea();

                if(float SAH = leftN * leftS + rightN * rightS; SAH < bestSAH){
                    bestSAH = SAH;
                    bestSize = itSize;
                }
            }
            printf("bestSAH = %f \t itSize = %lu \t bestSize = %lu\n", bestSAH, itSize, bestSize);
            printf("------------------------------------------------------\n");

            left = new kdNode(begin, bestSize);
            right = new kdNode(begin + left->size, size - left->size);
        }
    }

    __host__ __device__ kdNode(iterator begin, size_t size, uint32_t* offsets, size_t& nodesCounter) : begin(begin), size(size){
        sort(begin, size, box);

        if(size > 10)
        {
            uint32_t leftSize = offsets[nodesCounter++];
            left = new kdNode(begin, leftSize, offsets, nodesCounter);
            uint32_t rightSize = offsets[nodesCounter++];
            right = new kdNode(begin + left->size, rightSize, offsets, nodesCounter);
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

template <typename iterator>
size_t findMaxDepth(kdNode<iterator>* node, size_t& index){
    if(node){
        size_t res = ++index;
        res = std::max(res, findMaxDepth(node->left, index));
        res = std::max(res, findMaxDepth(node->right, index));
        --index;
        return res;
    }
    return 0;
}

template <typename iterator>
void buildSizesVector(kdNode<iterator>* node, std::vector<uint32_t>& nodeCounter){
    if(node){
        nodeCounter.push_back(node->size);
        buildSizesVector(node->left, nodeCounter);
        buildSizesVector(node->right, nodeCounter);
    }
}

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

    __host__ __device__ void makeTree(uint32_t* offsets) {
        size_t nodesCounter = 0;
        uint32_t size = offsets[nodesCounter++];
        root = new kdNode<container::iterator>(storage->begin(), size, offsets, nodesCounter);
    }

    __host__ __device__ iterator begin() {return storage->begin(); }
    __host__ __device__ iterator end() {return storage->end(); }

    __host__ __device__ iterator begin() const {return storage->begin(); }
    __host__ __device__ iterator end() const {return storage->end(); }

    static void create(kdTree* dpointer, const kdTree& host);
    static void destroy(kdTree* dpointer);
};

void makeTree(kdTree* container, uint32_t* offsets);
}

#endif // KDTREE_H
