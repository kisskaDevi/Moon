#include "models/model.h"

namespace cuda {

__global__ void deleteList(hitable* obj) {
    delete obj;
}

void destroy(hitable* obj) {
    deleteList <<<1, 1>>> (obj);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

model::~model(){
    for(auto obj: hitables){
        destroy(obj);
    }
}

model::model(const std::vector<vertex>& vertexBuffer, const std::vector<uint32_t>& indexBuffer)
    : vertexBuffer(cuda::buffer<vertex>(vertexBuffer.size(), vertexBuffer.data())) {
    for (size_t index = 0; index < indexBuffer.size(); index += 3) {
        triangle tr(indexBuffer[index + 0], indexBuffer[index + 1], indexBuffer[index + 2], vertexBuffer.data());
        boxes.push_back(tr.bbox);
        hitables.push_back(triangle::create(indexBuffer[index + 0], indexBuffer[index + 1], indexBuffer[index + 2], this->vertexBuffer.get()));
    }
}

model::model(const std::vector<hitable*>& hitables, const std::vector<box>& boxes)
    : hitables(hitables), boxes(boxes) {}

}
