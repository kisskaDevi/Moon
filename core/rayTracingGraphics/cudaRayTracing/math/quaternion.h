#ifndef QUATERNION_H
#define QUATERNION_H

#include "vec4.h"

namespace cuda::rayTracing {

template<typename type>
class Quaternion
{
private:
    type s;
    type x;
    type y;
    type z;

public:
    __host__ __device__ Quaternion();
    __host__ __device__ Quaternion(const Quaternion<type>& other);
    __host__ __device__ Quaternion(const type& s, const type& x, const type& y, const type& z);
    __host__ __device__ Quaternion(const type& s, const vec4<type>& v);
    __host__ __device__ Quaternion<type>& operator=(const Quaternion<type>& other);
    __host__ __device__ ~Quaternion();

    __host__ __device__ type                scalar()const;
    __host__ __device__ vec4<type>          vector()const;

    __host__ __device__ bool                operator==(const Quaternion<type>& other)const;
    __host__ __device__ bool                operator!=(const Quaternion<type>& other)const;
    __host__ __device__ Quaternion<type>    operator+ (const Quaternion<type>& other)const;
    __host__ __device__ Quaternion<type>    operator- (const Quaternion<type>& other)const;
    __host__ __device__ Quaternion<type>    operator* (const Quaternion<type>& other)const;
    __host__ __device__ Quaternion<type>&   operator+=(const Quaternion<type>& other);
    __host__ __device__ Quaternion<type>&   operator-=(const Quaternion<type>& other);
    __host__ __device__ Quaternion<type>&   operator*=(const Quaternion<type>& other);

    __host__ __device__ Quaternion<type>& normalize();
    __host__ __device__ Quaternion<type>& conjugate();
    __host__ __device__ Quaternion<type>& invert();

    template<typename T> __host__ __device__ friend Quaternion<T>   normalize(const Quaternion<T>& quat);
    template<typename T> __host__ __device__ friend Quaternion<T>   conjugate(const Quaternion<T>& quat);
    template<typename T> __host__ __device__ friend Quaternion<T>   invert(const Quaternion<T>& quat);

    template<typename T> __host__ __device__ friend Quaternion<T> operator* (const T& c, const Quaternion<T>& quat);
    template<typename T> friend std::ostream& operator<< (std::ostream& out, const Quaternion<T>& quat);

    //template<typename T> friend Quaternion<T> convert(const glm::mat<3, 3, T, glm::defaultp>& O3);
    //template<typename T> friend glm::mat3x3 convert(const Quaternion<T>& quat);

    template<typename T> __host__ __device__ friend Quaternion<T> convert(const T& yaw, const T& pitch, const T& roll);
    template<typename T> __host__ __device__ friend Quaternion<T> convert(const T& angle, const vec4<type>& axis);

    template<typename T> __host__ __device__ friend vec4<type> convertToEulerAngles(const Quaternion<T>& quat);
    template<typename T> __host__ __device__ friend Quaternion<T> convertToAnglesAndAxis(const Quaternion<T>& quat);

    template<typename T> __host__ __device__ friend Quaternion<T> slerp(const Quaternion<T>& quat1, const Quaternion<T>& quat2, const T& t);
};


template<typename type>
__host__ __device__ Quaternion<type>::Quaternion() :
    s(static_cast<type>(0)),
    x(static_cast<type>(0)),
    y(static_cast<type>(0)),
    z(static_cast<type>(0))
{}

template<typename type>
__host__ __device__ Quaternion<type>::Quaternion(const Quaternion<type>& other) :
    s(other.s),
    x(other.x),
    y(other.y),
    z(other.z)
{}

template<typename type>
__host__ __device__ Quaternion<type>::Quaternion(const type& s, const type& x, const type& y, const type& z) :
    s(s),
    x(x),
    y(y),
    z(z)
{}

template<typename type>
__host__ __device__ Quaternion<type>::Quaternion(const type& s, const vec4<type>& v) :
    s(s),
    x(static_cast<type>(v.x())),
    y(static_cast<type>(v.y())),
    z(static_cast<type>(v.z()))
{}

template<typename type>
__host__ __device__ Quaternion<type>& Quaternion<type>::operator=(const Quaternion<type>& other)
{
    s = other.s;
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
}

template<typename type>
__host__ __device__ Quaternion<type>::~Quaternion()
{}

template<typename type>
__host__ __device__ type Quaternion<type>::scalar()const
{
    return s;
}

template<typename type>
__host__ __device__ vec4<type> Quaternion<type>::vector()const
{
    return vec4<type>(x, y, z, static_cast<type>(0));
}

template<typename type>
__host__ __device__ bool Quaternion<type>::operator==(const Quaternion<type>& other)const
{
    return x == other.x && y == other.y && z == other.z && s == other.s;
}

template<typename type>
__host__ __device__ bool Quaternion<type>::operator!=(const Quaternion<type>& other)const
{
    return !(x == other.x && y == other.y && z == other.z && s == other.s);
}

template<typename type>
__host__ __device__ Quaternion<type> Quaternion<type>::operator+(const Quaternion<type>& other)const
{
    return Quaternion<type>(s + other.s, x + other.x, y + other.y, z + other.z);
}

template<typename type>
__host__ __device__ Quaternion<type>    Quaternion<type>::operator-(const Quaternion<type>& other)const
{
    return Quaternion<type>(s - other.s, x - other.x, y - other.y, z - other.z);
}

template<typename type>
__host__ __device__ Quaternion<type>    Quaternion<type>::operator*(const Quaternion<type>& other)const
{
    return Quaternion<type>(
        s * other.s - (x * other.x + y * other.y + z * other.z),
        s * other.x + other.s * x + (y * other.z - z * other.y),
        s * other.y + other.s * y + (z * other.x - x * other.z),
        s * other.z + other.s * z + (x * other.y - y * other.x)
        );
}

template<typename type>
__host__ __device__ Quaternion<type>& Quaternion<type>::operator+=(const Quaternion<type>& other)
{
    s += other.s;
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

template<typename type>
__host__ __device__ Quaternion<type>& Quaternion<type>::operator-=(const Quaternion<type>& other)
{
    s -= other.s;
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
}

template<typename type>
__host__ __device__ Quaternion<type>& Quaternion<type>::operator*=(const Quaternion<type>& other)
{
    Quaternion<type> copy(*this);
    *this = copy * other;

    return *this;
}

template<typename T>
std::ostream& operator<< (std::ostream& out, const Quaternion<T>& quat)
{
    out << quat.s << '\t' << quat.x << '\t' << quat.y << '\t' << quat.z;
    return out;
}

template<typename T>
__host__ __device__ Quaternion<T> operator* (const T& c, const Quaternion<T>& quat)
{
    return Quaternion<T>(c * quat.s, c * quat.x, c * quat.y, c * quat.z);
}

template<typename type>
__host__ __device__ Quaternion<type>& Quaternion<type>::normalize()
{
    type norma = s * s + x * x + y * y + z * z;
    norma = std::sqrt(norma);
    s /= norma;
    x /= norma;
    y /= norma;
    z /= norma;
    return *this;
}

template<typename type>
__host__ __device__ Quaternion<type>& Quaternion<type>::conjugate()
{
    x = -x;
    y = -y;
    z = -z;
    return *this;
}

template<typename type>
__host__ __device__ Quaternion<type>& Quaternion<type>::invert()
{
    Quaternion<type> quat(*this);
    Quaternion<type> ivNorma = quat * this->conjugate();
    ivNorma.s = std::sqrt(ivNorma.s);
    *this = ivNorma * (*this);
    return *this;
}

template<typename T>
__host__ __device__ Quaternion<T> normalize(const Quaternion<T>& quat)
{
    T norma = quat.s * quat.s + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z;
    norma = std::sqrt(norma);
    return Quaternion<T>(quat.s / norma, quat.x / norma, quat.y / norma, quat.z / norma);
}

template<typename T>
__host__ __device__ Quaternion<T> conjugate(const Quaternion<T>& quat)
{
    return Quaternion<T>(quat.s, -quat.x, -quat.y, -quat.z);
}

template<typename T>
__host__ __device__ Quaternion<T>   invert(const Quaternion<T>& quat)
{
    Quaternion<T> ivNorma = quat * conjugate(quat);
    ivNorma.s = std::sqrt(ivNorma.s);
    return ivNorma * conjugate(quat);
}

// template<typename T>
// Quaternion<T> convert(const glm::mat<3, 3, T, glm::defaultp>& O3)
// {
//     Quaternion<T> quat;
//
//     quat.s = std::sqrt(1.0f + O3[0][0] + O3[1][1] + O3[2][2]) / 2.0f;
//
//     quat.z = (O3[1][0] - O3[0][1]) / (T(4) * quat.s);
//     quat.y = (O3[0][2] - O3[2][0]) / (T(4) * quat.s);
//     quat.x = (O3[2][1] - O3[1][2]) / (T(4) * quat.s);
//
//     return quat;
// }
//
// template<typename T>
// glm::mat3x3 convert(const Quaternion<T>& quat)
// {
//     glm::mat3x3 R;
//
//     R[0][0] = T(1) - T(2) * (quat.y * quat.y + quat.z * quat.z);      R[0][1] = T(2) * (quat.x * quat.y - quat.z * quat.s);         R[0][2] = T(2) * (quat.x * quat.z + quat.y * quat.s);
//     R[1][0] = T(2) * (quat.x * quat.y + quat.z * quat.s);             R[1][1] = T(1) - T(2) * (quat.x * quat.x + quat.z * quat.z);  R[1][2] = T(2) * (quat.y * quat.z - quat.x * quat.s);
//     R[2][0] = T(2) * (quat.x * quat.z - quat.y * quat.s);             R[2][1] = T(2) * (quat.y * quat.z + quat.x * quat.s);         R[2][2] = T(1) - T(2) * (quat.x * quat.x + quat.y * quat.y);
//
//     return R;
// }

template<typename T>
__host__ __device__ Quaternion<T> convert(const T& yaw, const T& pitch, const T& roll)
{
    T cosy = std::cos(yaw * T(0.5));
    T siny = std::sin(yaw * T(0.5));
    T cosp = std::cos(pitch * T(0.5));
    T sinp = std::sin(pitch * T(0.5));
    T cosr = std::cos(roll * T(0.5));
    T sinr = std::sin(roll * T(0.5));

    T s = cosy * cosp * cosr + siny * sinp * sinr;
    T x = sinr * cosp * cosy - cosr * sinp * siny;
    T y = cosr * sinp * cosy + sinr * cosp * siny;
    T z = cosr * cosp * siny - sinr * sinp * cosy;

    return Quaternion<T>(s, x, y, z);
}

template<typename T>
__host__ __device__ Quaternion<T> convert(const T& angle, const vec4<T>& axis)
{
    return Quaternion<T>(std::cos(angle * T(0.5)), std::sin(angle * T(0.5)) * vec4<T>(axis.x(), axis.y(), axis.z(), T(0)));
}

template<typename T>
__host__ __device__ vec4<T> convertToEulerAngles(const Quaternion<T>& quat)
{
    return  vec4<T>(
        std::atan((quat.s * quat.x + quat.y * quat.z) * T(2) / (T(1) - (quat.x * quat.x + quat.y * quat.y) * T(2))),
        std::asin((quat.s * quat.y - quat.x * quat.z) * T(2)),
        std::atan((quat.s * quat.z + quat.y * quat.x) * T(2) / (T(1) - (quat.z * quat.z + quat.y * quat.y) * T(2))),
        T(0)
    );
}

template<typename T>
__host__ __device__ Quaternion<T> convertToAnglesAndAxis(const Quaternion<T>& quat)
{
    return Quaternion<T>(std::acos(quat.s) * T(2),
        vec4<T>(quat.x, quat.y, quat.z, T(0)) / std::sqrt(T(1) - quat.s * quat.s));
}

template<typename T>
__host__ __device__ Quaternion<T> slerp(const Quaternion<T>& quat1, const Quaternion<T>& quat2, const T& t)
{
    T q1q2 = quat1.s * quat2.s + quat1.x * quat2.x + quat1.y * quat2.y + quat1.z * quat2.z;
    T modq1 = std::sqrt(quat1.s * quat1.s + quat1.x * quat1.x + quat1.y * quat1.y + quat1.z * quat1.z);
    T modq2 = std::sqrt(quat2.s * quat2.s + quat2.x * quat2.x + quat2.y * quat2.y + quat2.z * quat2.z);
    T theta = std::acos(q1q2 / modq1 / modq2);

    return std::sin((T(1) - t) * theta) / std::sin(theta) * quat1 + std::sin(t * theta) / std::sin(theta) * quat2;
}

}

#endif // QUATERNION_H
