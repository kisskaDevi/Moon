#ifndef BUFFERH
#define BUFFERH

#include "operations.h"

namespace cuda {

    template <typename Type>
    class buffer
    {
    private:
        Type* memory{ nullptr };
        size_t size{ 0 };

    public:
        buffer() = default;
        buffer(const size_t& size) : memory(static_cast<Type*>(Buffer::create(sizeof(Type) * size))), size(size) {}

        buffer(const size_t& size, Type* mem) : memory(static_cast<Type*>(Buffer::create(sizeof(Type) * size))), size(size)
        {
            cudaMemcpy(memory, mem, size * sizeof(Type), cudaMemcpyHostToDevice);
            checkCudaErrors(cudaGetLastError());
        }

        void create(size_t size) {
            memory = static_cast<Type*>(Buffer::create(sizeof(Type) * size));
            this->size = size;
        }

        buffer(const buffer& other) = delete;
        buffer& operator=(const buffer& other) = delete;

        buffer(buffer&& other) : memory(other.memory), size(other.size)
        {
            other.memory = nullptr;
            other.size = 0;
        }

        buffer& operator=(buffer&& other)
        {
            memory = other.memory;
            size = other.size;
            other.memory = nullptr;
            other.size = 0;
            return *this;
        }

        void destroy() {
            if (memory) {
                checkCudaErrors(cudaFree(memory));
            }
            memory = nullptr;
            size = 0;
        }

        ~buffer() {
            destroy();
        }

        Type& operator[](size_t i) {
            return *(memory + i);
        }

        Type* get() {
            return memory;
        }
    };
}

#endif // !BUFFERH
