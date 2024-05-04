#include "utils/kdTree.h"
#include "utils/operations.h"

namespace cuda {

__global__ void createTreeKernel(hitableKDTree* tree, uint32_t* offsets)
{
    tree->makeTree(offsets);
}

void makeTree(hitableKDTree* container, uint32_t* offsets){
    createTreeKernel<<<1,1>>>(container, offsets);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void createKernel(hitableKDTree* p) {
    p = new (p) hitableKDTree();
}

void hitableKDTree::create(hitableKDTree* dpointer, const hitableKDTree& host){
    createKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(hitableKDTree* p) {
    p->~hitableKDTree();
}

void hitableKDTree::destroy(hitableKDTree* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
}
