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

__host__ __device__ bool hitableList::hit(const ray& r, hitCoords& coord) const {
    for(iterator it = begin(); it != end(); it++){
        if ((*it)->hit(r, coord)) {
            coord.obj = *it;
        }
    }
    return coord.obj;
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

__host__ __device__ hitable*&hitableList::operator[](uint32_t i) const {
    if(i == container_size - 1){
        return tail->current;
    }
    node* currentNode = head;
    for (; i > 0; i--) {
        currentNode = currentNode->next;
    }
    return currentNode->current;
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
