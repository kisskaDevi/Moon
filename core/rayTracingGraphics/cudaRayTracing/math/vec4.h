#ifndef VEC4H
#define VEC4H

#include <math.h>
#include <iostream>
#include <curand_kernel.h>

#define pi 3.14159265358979323846f

namespace cuda {

class vec4 {
    float e[4];

public:
    __host__ __device__ vec4() {}
    __host__ __device__ vec4(float e0, float e1, float e2, float e3) { e[0] = e0; e[1] = e1; e[2] = e2; e[3] = e3;}
    __host__ __device__ vec4(float e0) { e[0] = e0; e[1] = e0; e[2] = e0; e[3] = e0; }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float w() const { return e[3]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }
    __host__ __device__ inline float a() const { return e[3]; }

    __host__ __device__ inline const vec4& operator+() const { return *this; }
    __host__ __device__ inline vec4 operator-() const { return vec4(-x(), -y(), -z(), -w()); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline vec4& operator+=(const vec4& v2);
    __host__ __device__ inline vec4& operator-=(const vec4& v2);
    __host__ __device__ inline vec4& operator*=(const vec4& v2);
    __host__ __device__ inline vec4& operator/=(const vec4& v2);
    __host__ __device__ inline vec4& operator*=(const float t);
    __host__ __device__ inline vec4& operator/=(const float t);

    __host__ __device__ inline float length2() const { return x() * x() + y() * y() + z() * z() + w() * w(); }
    __host__ __device__ inline float length() const { return sqrt(length2()); }
    __host__ __device__ inline vec4& normalize() { return *this *= (1.0f / length());}

    __host__ __device__ static vec4 getHorizontal(const vec4& d) {
        float D = std::sqrt(d.x() * d.x() + d.y() * d.y());
        return D > 0.0f ? vec4(d.y() / D, -d.x() / D, 0.0f, 0.0f) : vec4(1.0f, 0.0, 0.0f, 0.0f);
    }

    __host__ __device__ static vec4 getVertical(const vec4& d) {
        float z = std::sqrt(d.x() * d.x() + d.y() * d.y());
        return z > 0.0f ? vec4(-d.z() * d.x() / z / d.length(), -d.z() * d.y() / z / d.length(), z, 0.0f) : vec4(0.0f, 1.0, 0.0f, 0.0f);
    }
};

__host__ inline std::ostream& operator<<(std::ostream& os, const vec4& t) {
    os << t.x() << '\t' << t.y() << '\t' << t.z() << '\t' << t.w();
    return os;
}

__host__ __device__ inline vec4 operator+(const vec4& v1, const vec4& v2) {
    return vec4(v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z(), v1.w() + v2.w());
}

__host__ __device__ inline vec4 operator-(const vec4& v1, const vec4& v2) {
    return vec4(v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z(), v1.w() - v2.w());
}

__host__ __device__ inline vec4 operator*(const vec4& v1, const vec4& v2) {
    return vec4(v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z(), v1.w() * v2.w());
}

__host__ __device__ inline vec4 operator/(const vec4& v1, const vec4& v2) {
    return vec4(v1.x() / v2.x(), v1.y() / v2.y(), v1.z() / v2.z(), v1.w() / v2.w());
}

__host__ __device__ inline vec4 operator*(float t, const vec4& v) {
    return vec4(t * v.x(), t * v.y(), t * v.z(), t * v.w());
}

__host__ __device__ inline vec4 operator/(vec4 v, float t) {
    return vec4(v.x() / t, v.y() / t, v.z() / t, v.w() / t);
}

__host__ __device__ inline vec4 operator*(const vec4& v, float t) {
    return vec4(t * v.x(), t * v.y(), t * v.z(), t * v.w());
}

__host__ __device__ inline float dot(const vec4& v1, const vec4& v2) {
    return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z() + v1.w() * v2.w();
}

__host__ __device__ inline vec4& vec4::operator+=(const vec4& v) {
    e[0] += v.x();
    e[1] += v.y();
    e[2] += v.z();
    e[3] += v.w();
    return *this;
}

__host__ __device__ inline vec4& vec4::operator*=(const vec4& v) {
    e[0] *= v.x();
    e[1] *= v.y();
    e[2] *= v.z();
    e[3] *= v.w();
    return *this;
}

__host__ __device__ inline vec4& vec4::operator/=(const vec4& v) {
    e[0] /= v.x();
    e[1] /= v.y();
    e[2] /= v.z();
    e[3] /= v.w();
    return *this;
}

__host__ __device__ inline vec4& vec4::operator-=(const vec4& v) {
    e[0] -= v.x();
    e[1] -= v.y();
    e[2] -= v.z();
    e[3] -= v.w();
    return *this;
}

__host__ __device__ inline vec4& vec4::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    e[3] *= t;
    return *this;
}

__host__ __device__ inline vec4& vec4::operator/=(const float t) {
    float k = 1.0f / t;
    *this *= k;
    return *this;
}

__host__ __device__ inline vec4 normal(const vec4& v) {
    return v / v.length();
}

__host__ __device__ inline vec4 cross(const vec4& v1, const vec4& v2) {
    return vec4(v1.y() * v2.z() - v1.z() * v2.y(), v1.z() * v2.x() - v1.x() * v2.z(), v1.x() * v2.y() - v1.y() * v2.x(), 0.0f);
}

__device__ inline vec4 random_in_unit_sphere(const vec4& direction, const float& angle, curandState* local_rand_state) {
    float phi = 2 * pi * curand_uniform(local_rand_state);
    float theta = angle * curand_uniform(local_rand_state);

    float x = std::sin(theta) * std::cos(phi);
    float y = std::sin(theta) * std::sin(phi);
    float z = std::cos(theta);

    return normal(x * vec4::getHorizontal(direction) + y * vec4::getVertical(direction) + z * direction);
}

__host__ __device__ inline vec4 max(const vec4& v1, const vec4& v2) {
    return vec4(v1.x() >= v2.x() ? v1.x() : v2.x(),
                v1.y() >= v2.y() ? v1.y() : v2.y(),
                v1.z() >= v2.z() ? v1.z() : v2.z(),
                v1.w() >= v2.w() ? v1.w() : v2.w());
}

__host__ __device__ inline vec4 min(const vec4& v1, const vec4& v2) {
    return vec4(v1.x() < v2.x() ? v1.x() : v2.x(),
                v1.y() < v2.y() ? v1.y() : v2.y(),
                v1.z() < v2.z() ? v1.z() : v2.z(),
                v1.w() < v2.w() ? v1.w() : v2.w());
}

__host__ __device__ inline float det3(const vec4& a, const vec4& b, const vec4& c) {
    return a.x() * b.y() * c.z() + b.x() * c.y() * a.z() + c.x() * a.y() * b.z() -
           (a.x() * c.y() * b.z() + b.x() * a.y() * c.z() + c.x() * b.y() * a.z());
}

}

#endif
