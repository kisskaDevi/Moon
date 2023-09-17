#include "hitableContainer.h"

#include "operations.h"

__global__ void addKernel(hitableContainer* container, hitable* object) {
    container->add(object);
}

void add(hitableContainer* container, std::vector<hitable*> objects) {
    for (auto& object : objects) {
        addKernel<<<1,1>>>(container, object);
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
