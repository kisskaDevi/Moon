#ifndef KDTREE_H
#define KDTREE_H

#include "hitableArray.h"
#include "utils/stack.h"

namespace cuda::rayTracing {

template <typename iterator>
__host__ __device__ box calcBox(iterator begin, iterator end){
    box resbox;
    for(auto it = begin; it != end; it++){
        const box itbox = (*it)->getBox();
        resbox.min = min(itbox.min, resbox.min);
        resbox.max = max(itbox.max, resbox.max);
    }
    return resbox;
}

template <typename iterator>
struct KDNode{
    KDNode* left{nullptr};
    KDNode* right{nullptr};
    iterator begin;
    size_t size{0};
    box bbox;

    static constexpr size_t itemsInNode = 10;
    static constexpr size_t stackSize = 50;

    __host__ __device__ KDNode(){}
    __host__ __device__ KDNode(const KDNode& other) = delete;
    __host__ __device__ KDNode& operator=(const KDNode& other) = delete;

    __host__ __device__ KDNode(KDNode&& other) : left(other.left), right(other.right), begin(other.begin), size(other.size), bbox(other.bbox) {
        other.left = nullptr; other.right = nullptr; other.size = 0;
    }
    __host__ __device__ KDNode& operator=(KDNode&& other)
    {
        left = other.left; right = other.right; begin = other.begin; size = other.size; bbox = other.bbox;
        other.left = nullptr; other.right = nullptr; other.size = 0;
        return *this;
    }

    __host__ __device__ iterator end() const { return begin + size;}

    __host__ KDNode(iterator begin, size_t size) : begin(begin), size(size), bbox(calcBox(begin, begin + size)){
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

            left = new KDNode(begin, bestSize);
            right = new KDNode(begin + bestSize, size - bestSize);
        }
    }

    __host__ __device__ ~KDNode() {
        if(left) delete left;
        if(right) delete right;
    }

    __host__ __device__ bool hit(const ray& r, HitCoords& coord) {
        Stack<const KDNode*, KDNode::stackSize> selected;
        for(Stack<const KDNode*, KDNode::stackSize> traversed(this); !traversed.empty();){
            if(const KDNode* curr = traversed.top(); traversed.pop() && curr->bbox.intersect(r)){
                if (!curr->left && !curr->right){
                    selected.push(curr);
                } else {
                    traversed.push(curr->left);
                    traversed.push(curr->right);
                }
            }
        }

        for(auto curr = selected.top(); selected.pop(); curr = selected.top()){
            for(iterator it = curr->begin; it != curr->end(); it++){
                if ((*it)->hit(r, coord)) coord.obj = *it;
            }
        }

        return coord.obj;
    }
};

template <typename iterator>
size_t findMaxDepth(KDNode<iterator>* node, size_t& index){
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
void buildSizesVector(KDNode<iterator>* node, std::vector<uint32_t>& linearSizes){
    if(node){
        linearSizes.push_back(node->size);
        buildSizesVector(node->left, linearSizes);
        buildSizesVector(node->right, linearSizes);
    }
}

template <typename iterator>
void buildBoxesVector(KDNode<iterator>* node, std::vector<box>& linearBoxes){
    if(node){
        linearBoxes.push_back(node->bbox);
        buildBoxesVector(node->left, linearBoxes);
        buildBoxesVector(node->right, linearBoxes);
    }
}

template <typename iterator>
void buildOffsetVector(KDNode<iterator>* node, std::vector<uint32_t>& linearOffsets, uint32_t& offset){
    if(node){
        linearOffsets.push_back(offset);
        if(node->size <= KDNode<iterator>::itemsInNode) {
            offset += node->size;
        }
        buildOffsetVector(node->left, linearOffsets, offset);
        buildOffsetVector(node->right, linearOffsets, offset);
    }
}

struct Nodes{
    std::vector<uint32_t> curr;
    std::vector<uint32_t> left;
    std::vector<uint32_t> right;
};

template <typename iterator>
void buildNodesVector(KDNode<iterator>* node, Nodes& nodes, uint32_t& counter, uint32_t curr){
    if(node){
        nodes.curr.push_back(curr);
        if(node->size > KDNode<iterator>::itemsInNode) {
            uint32_t right = counter++;
            uint32_t left = counter++;
            nodes.right.push_back(right);
            nodes.left.push_back(left);
            buildNodesVector(node->left, nodes, counter, left);
            buildNodesVector(node->right, nodes, counter, right);
        } else {
            nodes.right.push_back(counter);
            nodes.left.push_back(counter);
        }
    }
}

template <typename container>
class KDTree{
private:
    KDNode<typename container::iterator>* root{nullptr};
    size_t maxDepth{0};

public:
    container storage;

    __host__ KDTree() {}
    __host__ ~KDTree(){
        if(root) delete root;
    }
    __host__ void makeTree(){
        root = new KDNode<typename container::iterator>(storage.begin(), storage.size());
        maxDepth = findMaxDepth(root, maxDepth);
    }
    KDNode<typename container::iterator>* getRoot() const {
        return root;
    }
    std::vector<uint32_t> getLinearSizes() const {
        std::vector<uint32_t> linearSizes;
        buildSizesVector(root, linearSizes);
        return linearSizes;
    }
    std::vector<box> getLinearBoxes() const {
        std::vector<box> linearBoxes;
        buildBoxesVector(root, linearBoxes);
        return linearBoxes;
    }
    std::vector<uint32_t> getLinearOffsets() const {
        std::vector<uint32_t> linearOffsets;
        uint32_t offset = 0;
        buildOffsetVector(root, linearOffsets, offset);
        return linearOffsets;
    }
    Nodes buildLeftRight() const {
        Nodes nodes;
        uint32_t counter = 1;
        buildNodesVector(root, nodes, counter, 0);
        return nodes;
    }
};

class HitableKDTree : public HitableContainer {
public:
    using container = HitableArray;

private:
    container* storage{nullptr};
    KDNode<container::iterator>* root{nullptr};

public:
    using iterator = container::iterator;
    using KDNodeType = KDNode<container::iterator>;

    __host__ __device__ HitableKDTree() {
        storage = new container();
    };
    __host__ __device__ ~HitableKDTree(){
        if(storage) delete storage;
        if(root)    delete [] root;
    }

    __host__ __device__ void setRoot(KDNodeType* root){
        this->root = root;
    }

    __host__ __device__ bool hit(const ray& r, HitCoords& coord) const override{
        return root->hit(r, coord);
    }

    __host__ __device__ void add(Hitable** objects, size_t size = 1) override{
        storage->add(objects, size);
    }

    __host__ __device__ Hitable*& operator[](uint32_t i) const override{
        return (*storage)[i];
    }

    __host__ __device__ void makeTree(uint32_t* offsets, uint32_t* sizes, box* boxes, KDNodeType* nodes, uint32_t* current, uint32_t* left, uint32_t* right, size_t counter) {
        KDNode<container::iterator>* curr = &nodes[current[counter]];

        curr->begin = storage->begin() + offsets[counter];
        curr->size = sizes[counter];
        curr->bbox = boxes[counter];

        if(curr->size > KDNodeType::itemsInNode) {
            curr->right = &nodes[right[counter]];
            curr->left = &nodes[left[counter]];
        }
    }

    __host__ __device__ iterator begin() {return storage->begin(); }
    __host__ __device__ iterator end() {return storage->end(); }

    __host__ __device__ iterator begin() const {return storage->begin(); }
    __host__ __device__ iterator end() const {return storage->end(); }

    static void create(HitableKDTree* dpointer, const HitableKDTree& host);
    static void destroy(HitableKDTree* dpointer);
};

void makeTree(HitableKDTree* container, uint32_t* offsets, uint32_t* sizes, box* boxes, uint32_t* current, uint32_t* left, uint32_t* right, size_t size);

}
#endif // KDTREE_H
