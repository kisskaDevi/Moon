#include "utils/kdTree.h"

namespace cuda {

__global__ void createTreeKernel(kdTree* tree, uint32_t* offsets)
{
    tree->makeTree(offsets);
}

void makeTree(kdTree* container, uint32_t* offsets){
    createTreeKernel<<<1,1>>>(container, offsets);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void createKernel(kdTree* p) {
    p = new (p) kdTree();
}

void kdTree::create(kdTree* dpointer, const kdTree& host){
    createKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(kdTree* p) {
    p->~kdTree();
}

void kdTree::destroy(kdTree* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
}
