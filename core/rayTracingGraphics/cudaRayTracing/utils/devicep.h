#ifndef DEVICEP_H
#define DEVICEP_H

#include "utils/operations.h"

namespace cuda::rayTracing {

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
class Devicep{
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
    Devicep(type* other){
        del();
        pointer = other;
    }

    explicit Devicep(const type& host){
        del();
        pointer = create(host);
    }

    Devicep(size_t size){
        del();
        checkCudaErrors(cudaMalloc((void**)&pointer, size * sizeof(type)));
    }

    Devicep() = default;
    Devicep(const Devicep& other) = delete;
    Devicep& operator=(const Devicep& other) = delete;

    Devicep(Devicep&& other){
        del();
        pointer = other.pointer;
        other.pointer = nullptr;
    }

    Devicep& operator=(Devicep&& other){
        del();
        pointer = other.pointer;
        other.pointer = nullptr;
        return *this;
    }

    ~Devicep(){
        del();
    }

    type* get() const {
        return pointer;
    }

    type* operator()() const {
        return get();
    }

    type* release() {
        type* temp = pointer;
        pointer = nullptr;
        return temp;
    }
};

template<typename ptype, typename type>
Devicep<ptype> make_devicep(const type& host){
    return Devicep<ptype>(create(host));
}

template<typename type>
type to_host(const Devicep<type>& dp, size_t offset = 0){
    type host;
    checkCudaErrors(cudaMemcpy(&host, dp.get() + offset, sizeof(type), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    return host;
}

template<typename type>
void to_device(const type& host, const Devicep<type>& dp, size_t offset = 0){
    checkCudaErrors(cudaMemcpy(dp.get() + offset, &host, sizeof(type), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

}
#endif // DEVICEP_H
