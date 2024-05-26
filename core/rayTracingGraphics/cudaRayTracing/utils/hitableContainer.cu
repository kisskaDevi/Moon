#include "hitableContainer.h"

#include "operations.h"

namespace cuda::rayTracing {

__global__ void addKernel(HitableContainer* container, Hitable** object, size_t size) {
    container->add(object, size);
}

void add(HitableContainer* container, std::vector<Hitable*>& objects) {
    addKernel<<<1,1>>>(container, objects.data(), objects.size());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(HitableContainer* p) {
    p->~HitableContainer();
}

void HitableContainer::destroy(HitableContainer* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

}
