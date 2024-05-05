#ifndef BUFFERH
#define BUFFERH

#include "operations.h"
#include "devicep.h"

namespace cuda::rayTracing {

template <typename type>
class Buffer
{
private:
    Devicep<type> memory{ nullptr };
    size_t size{ 0 };

public:
    Buffer() = default;
    ~Buffer() = default;
    Buffer(const size_t& size, const type* mem = nullptr) : memory(size), size(size) {
        if(mem){
            cudaMemcpy(memory.get(), mem, size * sizeof(type), cudaMemcpyHostToDevice);
            checkCudaErrors(cudaGetLastError());
        }
    }

    Buffer(const Buffer& other) = delete;
    Buffer& operator=(const Buffer& other) = delete;

    Buffer(Buffer&& other) : memory(std::move(other.memory)), size(std::move(other.size)){}
    Buffer& operator=(Buffer&& other){
        memory = std::move(other.memory);
        size = std::move(other.size);
        return *this;
    }

    type* get() {return memory();}
    size_t getSize() {return size;}
};

}
#endif // !BUFFERH
