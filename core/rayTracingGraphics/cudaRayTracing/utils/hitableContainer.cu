#include "hitableContainer.h"

#include "operations.h"

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

__global__ void deleteList(hitableContainer* list) {
    delete list;
}

void destroy(hitableContainer* list) {
    deleteList <<<1, 1>>> (list);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
