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

__global__ void createArray(hitableArray** arr) {
    *arr = new hitableArray();
}

hitableArray* hitableArray::create() {
    hitableArray** array;
    checkCudaErrors(cudaMalloc((void**)&array, sizeof(hitableArray**)));

    createArray<<<1, 1>>>(array);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hitableArray* hostarr = nullptr;
    checkCudaErrors(cudaMemcpy(&hostarr, array, sizeof(hitableArray*), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(array));

    return hostarr;
}

}
