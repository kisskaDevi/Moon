#include "hitable.h"
#include "operations.h"

namespace cuda {

__global__ void destroyKernel(hitable* p) {
    p->~hitable();
}

void hitable::destroy(hitable* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
}
