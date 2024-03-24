#include "hitableContainer.h"

#include "operations.h"

namespace cuda {

__global__ void addKernel(hitableContainer* container, hitable* object) {
    container->add(object);
}

void add(hitableContainer* container, const std::vector<hitable*>& objects) {
    for (auto& object : objects) {
        addKernel<<<1,1>>>(container, object);
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(hitableContainer* p) {
    p->~hitableContainer();
}

void hitableContainer::destroy(hitableContainer* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

}
