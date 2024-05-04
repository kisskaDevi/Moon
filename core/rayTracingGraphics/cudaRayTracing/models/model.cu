#include "models/model.h"

namespace cuda {

model::~model(){
}

model::model(const std::vector<vertex>& vertexBuffer, const std::vector<uint32_t>& indexBuffer)
    : vertexBuffer(cuda::buffer<vertex>(vertexBuffer.size(), vertexBuffer.data()))
{
    for (size_t index = 0; index < indexBuffer.size(); index += 3) {
        triangle tr(indexBuffer[index + 0], indexBuffer[index + 1], indexBuffer[index + 2], vertexBuffer.data());
        primitives.push_back({
            make_devicep<cuda::hitable>(triangle(indexBuffer[index + 0], indexBuffer[index + 1], indexBuffer[index + 2], this->vertexBuffer.get())),
            tr.getBox()
        });
    }
}

model::model(std::vector<primitive>&& primitives)
    : primitives(std::move(primitives)) {}

model::model(primitive&& primitive){
    primitives.emplace_back(std::move(primitive));
}

model::model(model&& m) : vertexBuffer(std::move(m.vertexBuffer)), primitives(std::move(m.primitives))
{}

model& model::operator=(model&& m)
{
    primitives.clear();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    vertexBuffer = std::move(m.vertexBuffer);
    primitives = std::move(m.primitives);
    return *this;
}

}
