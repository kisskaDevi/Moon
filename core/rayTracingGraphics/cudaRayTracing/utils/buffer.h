#ifndef BUFFERH
#define BUFFERH

#include "operations.h"
#include "devicep.h"

namespace cuda {

    template <typename type>
    class buffer
    {
    private:
        devicep<type> memory{ nullptr };
        size_t size{ 0 };

    public:
        buffer() = default;
        ~buffer() = default;
        buffer(const size_t& size, const type* mem = nullptr) : memory(size), size(size) {
            if(mem){
                cudaMemcpy(memory.get(), mem, size * sizeof(type), cudaMemcpyHostToDevice);
                checkCudaErrors(cudaGetLastError());
            }
        }

        buffer(const buffer& other) = delete;
        buffer& operator=(const buffer& other) = delete;

        buffer(buffer&& other) : memory(std::move(other.memory)), size(std::move(other.size)){}
        buffer& operator=(buffer&& other){
            memory = std::move(other.memory);
            size = std::move(other.size);
            return *this;
        }

        type* get() {return memory();}
    };
}

#endif // !BUFFERH
