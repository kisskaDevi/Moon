#include "utils/kdTree.h"
#include "utils/operations.h"

namespace cuda::rayTracing {

__global__ void createTreeKernel(HitableKDTree* tree, uint32_t* offsets)
{
    tree->makeTree(offsets);
}

void makeTree(HitableKDTree* container, uint32_t* offsets){
    createTreeKernel<<<1,1>>>(container, offsets);
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
