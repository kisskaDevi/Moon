#ifndef KDTREE_H
#define KDTREE_H

#include "hitableArray.h"
#include "utils/stack.h"

namespace cuda{

template <typename iterator>
__host__ __device__ box calcBox(iterator begin, iterator end){
    box resbox;
    for(auto it = begin; it != end; it++){
        const box itbox = (*it)->getBox();
        resbox.min = cuda::min(itbox.min, resbox.min);
        resbox.max = cuda::max(itbox.max, resbox.max);
    }
    return resbox;
}

template <typename iterator>
struct kdNode{
    kdNode* left{nullptr};
    kdNode* right{nullptr};
    iterator begin;
    size_t size{0};
    box bbox;

    static constexpr size_t itemsInNode = 10;
    static constexpr size_t stackSize = 50;

    __host__ __device__ kdNode(){}
    __host__ __device__ kdNode(const kdNode& other) = delete;
    __host__ __device__ kdNode& operator=(const kdNode& other) = delete;

    __host__ __device__ kdNode(kdNode&& other) : left(other.left), right(other.right), begin(other.begin), size(other.size), bbox(other.bbox) {
        other.left = nullptr; other.right = nullptr; other.size = 0;
    }
    __host__ __device__ kdNode& operator=(kdNode&& other)
    {
        left = other.left; right = other.right; begin = other.begin; size = other.size; bbox = other.bbox;
        other.left = nullptr; other.right = nullptr; other.size = 0;
        return *this;
    }

    __host__ kdNode(iterator begin, size_t size) : begin(begin), size(size), bbox(calcBox(begin, begin + size)){
        if(const iterator end = begin + size; size > itemsInNode)
        {
            sortByBox(begin, end, bbox);

            float bestSAH = std::numeric_limits<float>::max();
            size_t bestSize = 0, itSize = 0;
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

            left = new kdNode(begin, bestSize);
            right = new kdNode(begin + bestSize, size - bestSize);
        }
    }

    __host__ __device__ ~kdNode() {
        if(left) delete left;
        if(right) delete right;
    }

    __host__ __device__ bool hit(const ray& r, hitCoords& coord) const {
        stack<const kdNode*, kdNode::stackSize> selected;
        for(stack<const kdNode*, kdNode::stackSize> treeTraverse(this); !treeTraverse.empty();){
            const kdNode* curr = treeTraverse.top();
            treeTraverse.pop();

            if(curr->bbox.intersect(r)){
                if(curr->left)                      treeTraverse.push(curr->left);
                if(curr->right)                     treeTraverse.push(curr->right);
                if(!(curr->left || curr->right))    selected.push(curr);
            }
        }
        for(;!selected.empty();){
            const kdNode* curr = selected.top();
            selected.pop();
            for(iterator it = curr->begin; it != (curr->begin + curr->size); it++){
                if ((*it)->hit(r, coord))
                    coord.obj = *it;
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
void buildSizesVector(kdNode<iterator>* node, std::vector<uint32_t>& linearSizes){
    if(node){
        linearSizes.push_back(node->size);
        buildSizesVector(node->left, linearSizes);
        buildSizesVector(node->right, linearSizes);
    }
}

template <typename container>
class kdTree{
private:
    kdNode<typename container::iterator>* root{nullptr};
    size_t maxDepth{0};

public:
    container storage;

    __host__ kdTree() {}
    __host__ ~kdTree(){
        if(root) delete root;
    }
    __host__ void makeTree(){
        root = new kdNode<typename container::iterator>(storage.begin(), storage.size());
        maxDepth = findMaxDepth(root, maxDepth);
    }
    kdNode<typename container::iterator>* getRoot() const {
        return root;
    }
    std::vector<uint32_t> getLinearSizes() const {
        std::vector<uint32_t> linearSizes;
        buildSizesVector(root, linearSizes);
        return linearSizes;
    }
};

class hitableKDTree : public hitableContainer {
public:
    using container = hitableArray;

private:
    container* storage{nullptr};
    kdNode<container::iterator>* root{nullptr};

public:
    using iterator = container::iterator;

    __host__ __device__ hitableKDTree() {
        storage = new container();
    };
    __host__ __device__ ~hitableKDTree(){
        if(storage) delete storage;
        if(root)    delete root;
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

    __host__ __device__ void makeTree(uint32_t* offsets) {
        size_t nodesCounter = 0, offset = 0;

        for(stack<kdNode<container::iterator>*, kdNode<container::iterator>::stackSize> makeStack(root = new kdNode<container::iterator>()); !makeStack.empty();){
            kdNode<container::iterator>* curr = makeStack.top();
            makeStack.pop();

            curr->begin = storage->begin() + offset;
            curr->size = offsets[nodesCounter++];
            curr->bbox = calcBox(curr->begin, curr->begin + curr->size);

            if(curr->size > kdNode<container::iterator>::itemsInNode) {
                makeStack.push(curr->right = new kdNode<container::iterator>());
                makeStack.push(curr->left = new kdNode<container::iterator>());
            } else {
                offset += curr->size;
            }
        }
    }

    __host__ __device__ iterator begin() {return storage->begin(); }
    __host__ __device__ iterator end() {return storage->end(); }

    __host__ __device__ iterator begin() const {return storage->begin(); }
    __host__ __device__ iterator end() const {return storage->end(); }

    static void create(hitableKDTree* dpointer, const hitableKDTree& host);
    static void destroy(hitableKDTree* dpointer);
};

void makeTree(hitableKDTree* container, uint32_t* offsets);
}

#endif // KDTREE_H
