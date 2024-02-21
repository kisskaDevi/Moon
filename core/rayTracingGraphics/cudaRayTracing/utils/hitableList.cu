#include "hitableList.h"
#include "operations.h"

__host__ __device__ void destroyObject(hitable* object) {
    if (object->next) {
        destroyObject(object->next);
    }
    delete object;
}

__host__ __device__ hitableList::~hitableList() {
    destroyObject(head);
}

__device__ bool hitableList::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    float depth = tMax;
    hitCoords coord;
    hitable* resObj = nullptr;
    for (hitable* object = head; object; object = object->next) {
        if (object->hit(r, tMin, depth, coord)) {
            depth = coord.t;
            resObj = object;
        }
    }
    if(depth != tMax && resObj){
        resObj->calcHitRecord(r, coord, rec);
    }
    return depth != tMax;
}

__host__ __device__ void hitableList::add(hitable* object) {
    if (head) {
        tail->next = object;
    } else {
        head = object;
        head->next = object;
    }
    tail = object;
}

__global__ void createList(hitableList** list) {
    *list = new hitableList();
}

hitableList* hitableList::create() {
    hitableList** list;
    checkCudaErrors(cudaMalloc((void**)&list, sizeof(hitableList**)));

    createList<<<1, 1>>>(list);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hitableList* hostlist = nullptr;
    checkCudaErrors(cudaMemcpy(&hostlist, list, sizeof(hitableList*), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(list));

    return hostlist;
}