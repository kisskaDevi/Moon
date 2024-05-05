#include "hitableList.h"
#include "operations.h"

namespace cuda::rayTracing {

__host__ __device__ HitableList::~HitableList() {
    if(container_size){
        for (Node* next = head->next; next; container_size--, next = head->next) {
            delete head;
            head = next;
        }
    }
}

__host__ __device__ bool HitableList::hit(const ray& r, HitCoords& coord) const {
    for(iterator it = begin(); it != end(); it++){
        if ((*it)->hit(r, coord)) {
            coord.obj = *it;
        }
    }
    return coord.obj;
}

__host__ __device__ void HitableList::add(Hitable* object) {
    if(Node* newNode = new Node{object, nullptr}; tail){
        tail->next = newNode;
        tail = newNode;
    } else {
        head = newNode;
        tail = newNode;
    }
    container_size++;
}

__host__ __device__ Hitable*& HitableList::operator[](uint32_t i) const {
    if(i == container_size - 1){
        return tail->current;
    }
    Node* currentNode = head;
    for (; i > 0; i--) {
        currentNode = currentNode->next;
    }
    return currentNode->current;
}

__global__ void createKernel(HitableList* p) {
    p = new (p) HitableList();
}

void HitableList::create(HitableList* dpointer, const HitableList& host){
    createKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(HitableList* p) {
    p->~HitableList();
}

void HitableList::destroy(HitableList* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

}
