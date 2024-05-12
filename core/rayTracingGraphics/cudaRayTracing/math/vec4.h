#ifndef VEC4H
#define VEC4H

#include <math.h>
#include <curand_kernel.h>

#include <iostream>

#define pi 3.14159265358979323846f

namespace cuda::rayTracing {

template<typename T>
class vec4 {
    T e[4];

public:
    __host__ __device__ vec4<T>() {}
    __host__ __device__ vec4<T>(T e0, T e1, T e2, T e3) { e[0] = e0; e[1] = e1; e[2] = e2; e[3] = e3;}
    __host__ __device__ vec4<T>(T e0) { e[0] = e0; e[1] = e0; e[2] = e0; e[3] = e0; }
    __host__ __device__ T x() const { return e[0]; }
    __host__ __device__ T y() const { return e[1]; }
    __host__ __device__ T z() const { return e[2]; }
    __host__ __device__ T w() const { return e[3]; }
    __host__ __device__ T r() const { return e[0]; }
    __host__ __device__ T g() const { return e[1]; }
    __host__ __device__ T b() const { return e[2]; }
    __host__ __device__ T a() const { return e[3]; }

    __host__ __device__ const vec4<T>& operator+() const { return *this; }
    __host__ __device__ vec4<T> operator-() const { return vec4<T>(-x(), -y(), -z(), -w()); }
    __host__ __device__ T operator[](int i) const { return e[i]; }
    __host__ __device__ T& operator[](int i) { return e[i]; };

    __host__ __device__ vec4<T>& operator+=(const vec4<T>& v){
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        e[3] += v.e[3];
        return *this;
    }

    __host__ __device__ vec4<T>& operator*=(const vec4<T>& v){
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        e[3] *= v.e[3];
        return *this;
    }

    __host__ __device__ vec4<T>& operator*=(const T t){
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        e[3] *= t;
        return *this;
    }

    __host__ __device__ vec4<T>& operator-=(const vec4<T>& v){
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        e[3] -= v.e[3];
        return *this;
    }

    __host__ __device__ vec4<T>& operator/=(const vec4<T>& v){
        e[0] /= v.e[0];
        e[1] /= v.e[1];
        e[2] /= v.e[2];
        e[3] /= v.e[3];
        return *this;
    }

    __host__ __device__ vec4<T>& operator/=(const T t){
        e[0] /= t;
        e[1] /= t;
        e[2] /= t;
        e[3] /= t;
        return *this;
    }

    __host__ __device__ T length2() const { return x() * x() + y() * y() + z() * z() + w() * w(); }
    __host__ __device__ T length() const { return sqrt(length2()); }
    __host__ __device__ vec4<T>& normalize() { return *this *= (1.0f / length());}

    __host__ __device__ static vec4<T> getHorizontal(const vec4<T>& d) {
        T D = std::sqrt(d.x() * d.x() + d.y() * d.y());
        return D > 0.0f ? vec4<T>(d.y() / D, -d.x() / D, 0.0f, 0.0f) : vec4<T>(1.0f, 0.0, 0.0f, 0.0f);
    }

    __host__ __device__ static vec4<T> getVertical(const vec4<T>& d) {
        T z = std::sqrt(d.x() * d.x() + d.y() * d.y());
        return z > 0.0f ? vec4<T>(-d.z() * d.x() / z / d.length(), -d.z() * d.y() / z / d.length(), z, 0.0f) : vec4<T>(0.0f, 1.0, 0.0f, 0.0f);
    }

    __host__ __device__ size_t maxValueIndex(size_t lessThen = 4) const {
        size_t idx = 0;
        T v = e[idx];
        for(size_t i = 1; i < lessThen; i++)
            if(e[i] > v) v = e[idx = i];
        return idx;
    }

    __host__ __device__ size_t minValueIndex(size_t lessThen = 4) const {
        size_t idx = 0;
        T v = e[idx];
        for(size_t i = 1; i < lessThen; i++)
            if(e[i] < v) v = e[idx = i];
        return idx;
    }

};

template<typename T>
std::ostream& operator<<(std::ostream& os, const vec4<T>& t) {
    os << t.x() << '\t' << t.y() << '\t' << t.z() << '\t' << t.w();
    return os;
}

template<typename T>
__host__ __device__ inline vec4<T> operator+(const vec4<T>& v1, const vec4<T>& v2) {
    vec4<T> copy(v1);
    return copy += v2;
}

template<typename T>
__host__ __device__ vec4<T> operator-(const vec4<T>& v1, const vec4<T>& v2) {
    vec4<T> copy(v1);
    return copy -= v2;
}

template<typename T>
__host__ __device__ vec4<T> operator*(const vec4<T>& v1, const vec4<T>& v2) {
    vec4<T> copy(v1);
    return copy *= v2;
}

template<typename T>
__host__ __device__ vec4<T> operator/(const vec4<T>& v1, const vec4<T>& v2) {
    vec4<T> copy(v1);
    return copy /= v2;
}

template<typename T>
__host__ __device__ vec4<T> operator/(vec4<T> v, T t) {
    vec4<T> copy(v);
    return copy /= t;
}

template<typename T>
__host__ __device__ vec4<T> operator*(T t, const vec4<T>& v) {
    vec4<T> copy(v);
    return copy *= t;
}

template<typename T>
__host__ __device__ vec4<T> operator*(const vec4<T>& v, T t) {
    return t * v;
}

template<typename T>
__host__ __device__ T dot(const vec4<T>& v1, const vec4<T>& v2) {
    return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z() + v1.w() * v2.w();
}

template<typename T>
__host__ __device__ vec4<T> normal(const vec4<T>& v) {
    return v / v.length();
}

template<typename T>
__host__ __device__ inline vec4<T> cross(const vec4<T>& v1, const vec4<T>& v2) {
    return vec4<T>(v1.y() * v2.z() - v1.z() * v2.y(), v1.z() * v2.x() - v1.x() * v2.z(), v1.x() * v2.y() - v1.y() * v2.x(), 0.0f);
}

template<typename T>
__device__ inline vec4<T> random_in_unit_sphere(const vec4<T>& direction, const T& angle, curandState* randState) {
    T phi = 2 * pi * curand_uniform(randState);
    T theta = angle * curand_uniform(randState);

    T x = std::sin(theta) * std::cos(phi);
    T y = std::sin(theta) * std::sin(phi);
    T z = std::cos(theta);

    return normal(x * vec4<T>::getHorizontal(direction) + y * vec4<T>::getVertical(direction) + z * direction);
}

template<typename T>
__host__ __device__ inline vec4<T> max(const vec4<T>& v1, const vec4<T>& v2) {
    return vec4<T>( v1.x() >= v2.x() ? v1.x() : v2.x(),
                    v1.y() >= v2.y() ? v1.y() : v2.y(),
                    v1.z() >= v2.z() ? v1.z() : v2.z(),
                    v1.w() >= v2.w() ? v1.w() : v2.w());
}

template<typename T>
__host__ __device__ inline vec4<T> min(const vec4<T>& v1, const vec4<T>& v2) {
    return vec4<T>( v1.x() < v2.x() ? v1.x() : v2.x(),
                    v1.y() < v2.y() ? v1.y() : v2.y(),
                    v1.z() < v2.z() ? v1.z() : v2.z(),
                    v1.w() < v2.w() ? v1.w() : v2.w());
}

template<typename T>
__host__ __device__ inline T det3(const vec4<T>& a, const vec4<T>& b, const vec4<T>& c) {
    return a.x() * b.y() * c.z() + b.x() * c.y() * a.z() + c.x() * a.y() * b.z() -
           (a.x() * c.y() * b.z() + b.x() * a.y() * c.z() + c.x() * b.y() * a.z());
}

using vec4f = vec4<float>;
}
#endif
