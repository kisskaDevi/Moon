#ifndef OBJECTH
#define OBJECTH

#include "buffer.h"
#include "hitableContainer.h"
#include "material.h"
#include "triangle.h"

#include <vector>

struct primitive {
    cuda::buffer<vertex> vertexBuffer;
    std::vector<uint32_t> indexBuffer;
    size_t firstIndex{ 0 };
    size_t size{ 0 };

    primitive(
        std::vector<vertex>&& vertexBuffer,
        std::vector<uint32_t>&& indexBuffer,
        size_t firstIndex
    ) : vertexBuffer(std::move(cuda::buffer<vertex>(vertexBuffer.size(), vertexBuffer.data()))), indexBuffer(std::move(indexBuffer)), firstIndex(firstIndex), size(this->indexBuffer.size()) {}

    void moveToContainer(hitableContainer* container) {
        for (size_t index = firstIndex; index < size; index += 3) {
            add(container, {triangle::create(indexBuffer[index + 0], indexBuffer[index + 1], indexBuffer[index + 2], vertexBuffer.get())});
        }
    }
};

#endif // !OBJECTH
