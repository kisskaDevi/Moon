#ifndef STACK_H
#define STACK_H

#include <stddef.h>

namespace cuda{

template <typename type, size_t size = 20>
class stack {
public:

private:
    size_t container_size{0};
    type static_storage[size];

public:
    __host__ __device__ stack(){}
    __host__ __device__ ~stack(){}
    __host__ __device__ stack(const type& data){ push(data); }
    __host__ __device__ size_t fill() const { return container_size; }
    __host__ __device__ bool empty() const { return container_size == 0; }

    __host__ __device__ bool push(const type& data){
        if(container_size >= size){
            return false;
        }

        static_storage[container_size] = data;
        container_size++;
        return true;
    }

    __host__ __device__ type& top() {
        return static_storage[container_size - 1];
    }

    __host__ __device__ const type& top() const {
        return static_storage[container_size - 1];
    }

    __host__ __device__ bool pop(){
        if(container_size == 0){
            return false;
        }
        container_size--;
        return true;
    }
};

}

#endif // STACK_H
