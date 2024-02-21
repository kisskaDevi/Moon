#include "hitableArray.h"

#include "operations.h"

__host__ __device__ hitableArray::~hitableArray() {
    for(size_t i = 0; i < size; i++){
        delete array[i];
    }
    delete[] array;
    size = 0;
}

__device__ bool hitableArray::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    float depth = tMax;
    hitCoords coord;
    hitable* resObj = nullptr;
    for (size_t i = 0; i < size; i++) {
        if (array[i]->hit(r, tMin, depth, coord)) {
            depth = coord.t;
            resObj = array[i];
        }
    }
    if(depth != tMax && resObj){
        resObj->calcHitRecord(r, coord, rec);
    }
    return depth != tMax;
}

__host__ __device__ void hitableArray::add(hitable* object) {
    hitable** newArray = new hitable*[size + 1];
    for(size_t i = 0; i < size; i++){
        newArray[i] = array[i];
    }
    newArray[size] = object;
    delete[] array;
    array = newArray;
    size++;
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