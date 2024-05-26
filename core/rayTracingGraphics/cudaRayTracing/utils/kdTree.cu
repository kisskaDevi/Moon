#include "utils/kdTree.h"
#include "utils/operations.h"
#include "utils/buffer.h"

namespace cuda::rayTracing {

__global__ void setRootKernel(HitableKDTree* tree, HitableKDTree::KDNodeType* nodes)
{
    tree->setRoot(&nodes[0]);
}

__global__ void createTreeKernel(HitableKDTree* tree, uint32_t* offsets, uint32_t* sizes, box* boxes, uint32_t* current, uint32_t* left, uint32_t* right, HitableKDTree::KDNodeType* nodes)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    tree->makeTree(offsets, sizes, boxes, nodes, current, left, right, i);
}

void makeTree(HitableKDTree* container, uint32_t* offsets, uint32_t* sizes, box* boxes, uint32_t* current, uint32_t* left, uint32_t* right, size_t size){
    Buffer<HitableKDTree::KDNodeType> nodes(size);
    setRootKernel<<<1,1>>>(container, nodes.get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    createTreeKernel<<<size,1>>>(container, offsets, sizes, boxes, current, left, right, nodes.release());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void createKernel(HitableKDTree* p) {
    p = new (p) HitableKDTree();
}

void HitableKDTree::create(HitableKDTree* dpointer, const HitableKDTree& host){
    createKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(HitableKDTree* p) {
    p->~HitableKDTree();
}

void HitableKDTree::destroy(HitableKDTree* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

}
