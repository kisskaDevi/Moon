#include "hitableContainer.h"

#include "operations.h"

namespace cuda::rayTracing {

__global__ void addKernel(HitableContainer* container, Hitable* object) {
    container->add(object);
}

void add(HitableContainer* container, const std::vector<Hitable*>& objects) {
    for (auto& object : objects) {
        addKernel<<<1,1>>>(container, object);
    }
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
