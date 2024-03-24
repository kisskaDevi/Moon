#include "hitableList.h"
#include "operations.h"

namespace cuda {

__host__ __device__ hitableList::~hitableList() {
    if(container_size){
        for (node* next = head->next; next; container_size--, next = head->next) {
            delete head;
            head = next;
        }
    }
}

__host__ __device__ bool hitableList::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    hitCoords coord = {tMax, 0.0f, 0.0f};
    hitable* resObj = nullptr;
    for (node* currentNode = head; currentNode; currentNode = currentNode->next) {
        if (currentNode->current->hit(r, tMin, coord.t, coord)) {
            resObj = currentNode->current;
        }
    }
    if(coord.t != tMax && resObj){
        resObj->calcHitRecord(r, coord, rec);
        return true;
    }
    return false;
}

__host__ __device__ void hitableList::add(hitable* object) {
    if(node* newNode = new node{object, nullptr}; tail){
        tail->next = newNode;
        tail = newNode;
    } else {
        head = newNode;
        tail = newNode;
    }
    container_size++;
}

__host__ __device__ hitable* hitableList::operator[](uint32_t i) {
    if(i < container_size){
        node* currentNode = head;
        for (; i > 0; i--) {
            currentNode = currentNode->next;
        }
        return currentNode->current;
    }
    return nullptr;
}

__global__ void createList(hitableList** list) {
    *list = new hitableList();
}

__global__ void createKernel(hitableList* p) {
    p = new (p) hitableList();
}

void hitableList::create(hitableList* dpointer, const hitableList& host){
    createKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(hitableList* p) {
    p->~hitableList();
}

void hitableList::destroy(hitableList* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

}
