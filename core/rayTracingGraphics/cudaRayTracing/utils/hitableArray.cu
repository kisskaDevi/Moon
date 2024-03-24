#include "hitableArray.h"

#include "operations.h"

namespace cuda {

__host__ __device__ hitableArray::~hitableArray() {
    delete[] array;
    container_size = 0;
}

__host__ __device__ bool hitableArray::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    hitCoords coord = {tMax, 0.0f, 0.0f};
    hitable* resObj = nullptr;
    for (size_t i = 0; i < container_size; i++) {
        if (array[i]->hit(r, tMin, coord.t, coord)) {
            resObj = array[i];
        }
    }
    if(coord.t != tMax && resObj){
        resObj->calcHitRecord(r, coord, rec);
        return true;
    }
    return false;
}

__host__ __device__ void hitableArray::add(hitable* object) {
    hitable** newArray = new hitable*[container_size + 1];
    for(size_t i = 0; i < container_size; i++){
        newArray[i] = array[i];
    }
    newArray[container_size] = object;
    delete[] array;
    array = newArray;
    container_size++;
}

__host__ __device__ hitable* hitableArray::operator[](uint32_t i) {
    if(i >= container_size){
        return nullptr;
    }
    return array[i];
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
