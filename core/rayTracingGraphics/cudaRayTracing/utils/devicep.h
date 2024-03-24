#ifndef DEVICEP_H
#define DEVICEP_H

#include "utils/operations.h"

namespace cuda {

namespace{
template<typename T>
struct has_create
{
    template<typename U, void (*)(T*, const T&)> struct SFINAE {};
    template<typename U> static char Test(SFINAE<U, &U::create>*);
    template<typename U> static int Test(...);
    static constexpr bool v = sizeof(Test<T>(0)) == sizeof(char);
};

template<typename T>
struct has_destroy
{
    template<typename U, void (*)(T*)> struct SFINAE {};
    template<typename U> static char Test(SFINAE<U, &U::destroy>*);
    template<typename U> static int Test(...);
    static constexpr bool v = sizeof(Test<T>(0)) == sizeof(char);
};

template<typename type>
type* create(const type& host){
    type* pointer;
    checkCudaErrors(cudaMalloc((void**)&pointer, sizeof(type)));
    if constexpr (has_create<type>::v){
        type::create((type*)pointer, host);
    } else {
        checkCudaErrors(cudaMemcpy(pointer, &host, sizeof(type), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
    return pointer;
}
}

template<typename type>
class devicep{
private:
    type* pointer{nullptr};

    void del(){
        if(pointer){
            if constexpr (has_destroy<type>::v){
                type::destroy(pointer);
            }
            checkCudaErrors(cudaFree((void*)pointer));
            pointer = nullptr;
        }
    }
public:
    devicep(type* other){
        del();
        pointer = other;
    }

    devicep(const type& host){
        del();
        pointer = create(host);
    }

    devicep(size_t size){
        del();
        checkCudaErrors(cudaMalloc((void**)&pointer, size * sizeof(type)));
    }

    devicep() = default;
    devicep(const devicep& other) = delete;
    devicep& operator=(const devicep& other) = delete;

    devicep(devicep&& other){
        del();
        pointer = other.pointer;
        other.pointer = nullptr;
    }

    devicep& operator=(devicep&& other){
        del();
        pointer = other.pointer;
        other.pointer = nullptr;
        return *this;
    }

    ~devicep(){
        del();
    }

    type* get() const {
        return pointer;
    }

    type* operator()() const {
        return get();
    }
};

template<typename ptype, typename type>
devicep<ptype> make_devicep(const type& host){
    return devicep<ptype>(create(host));
}

template<typename type>
type to_host(const devicep<type>& dp, size_t offset = 0){
    type host;
    checkCudaErrors(cudaMemcpy(&host, dp.get() + offset, sizeof(type), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    return host;
}

template<typename type>
void to_device(const type& host, const devicep<type>& dp, size_t offset = 0){
    checkCudaErrors(cudaMemcpy(dp.get() + offset, &host, sizeof(type), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
}

#endif // DEVICEP_H
