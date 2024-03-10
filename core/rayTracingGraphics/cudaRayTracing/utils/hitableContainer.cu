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

__global__ void deleteContainer(hitableContainer* container) {
    delete container;
}

void destroy(hitableContainer* container) {
    deleteContainer <<<1, 1>>> (container);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

}
