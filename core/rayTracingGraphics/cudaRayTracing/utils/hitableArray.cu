#include "hitableArray.h"

#include "operations.h"

namespace cuda {

__host__ __device__ hitableArray::~hitableArray() {
    delete[] array;
    container_size = 0;
}

__host__ __device__ bool hitableArray::hit(const ray& r, hitCoords& coord) const {
    for(iterator it = begin(); it != end(); it++){
        if ((*it)->hit(r, coord)) {
            coord.obj = *it;
        }
    }
    return coord.obj;
}

__host__ __device__ void hitableArray::add(hitable* object) {
    pointer* newArray = new pointer[container_size + 1];
    for(size_t i = 0; i < container_size; i++){
        newArray[i] = array[i];
    }
    newArray[container_size].p = object;
    delete[] array;
    array = newArray;
    container_size++;
}

__host__ __device__ hitable*& hitableArray::operator[](uint32_t i) const {
    return array[i].p;
}

__global__ void createKernel(hitableArray* p) {
    p = new (p) hitableArray();
}

void hitableArray::create(hitableArray* dpointer, const hitableArray& host){
    createKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(hitableArray* p) {
    p->~hitableArray();
}

void hitableArray::destroy(hitableArray* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

}
