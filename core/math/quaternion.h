#ifndef QUATERNION_H
#define QUATERNION_H

#include <iostream>
#include "matrix.h"
#include <math.h>

template<typename type>
class quaternion
{
private:
    type s;
    type x;
    type y;
    type z;

public:
    quaternion();
    quaternion(const quaternion<type>& other);
    quaternion(const type& s,const type& x,const type& y,const type& z);
    quaternion(const type& s,const vector<type, 3>& v);
    quaternion<type>& operator=(const quaternion<type>& other);
    ~quaternion();

    type                scalar()const;
    vector<type, 3>    vector()const;

    bool                operator==(const quaternion<type>& other)const;
    bool                operator!=(const quaternion<type>& other)const;
    quaternion<type>    operator+ (const quaternion<type>& other)const;
    quaternion<type>    operator- (const quaternion<type>& other)const;
    quaternion<type>    operator* (const quaternion<type>& other)const;
    quaternion<type>&   operator+=(const quaternion<type>& other);
    quaternion<type>&   operator-=(const quaternion<type>& other);
    quaternion<type>&   operator*=(const quaternion<type>& other);

    quaternion<type>&   normalize();
    quaternion<type>&   conjugate();
    quaternion<type>&   invert();

    template<typename T> friend quaternion<T>   normalize(const quaternion<T>& quat);
    template<typename T> friend quaternion<T>   conjugate(const quaternion<T>& quat);
    template<typename T> friend quaternion<T>   invert(const quaternion<T>& quat);

    template<typename T> friend quaternion<T> operator* (const T& c, const quaternion<T>& quat);
    template<typename T> friend std::ostream& operator<< (std::ostream & out, const quaternion<T>& quat);

    template<typename T> friend quaternion<T> convert(const matrix<T,3,3>& O3);
    template<typename T> friend matrix<T,3,3> convert(const quaternion<T>& quat);
    template<typename T> friend matrix<T,4,4> convert4x4(const quaternion<T>& quat);

    template<typename T> friend quaternion<T> convert(const T& yaw, const T& pitch, const T& roll);
    template<typename T> friend quaternion<T> convert(const T& angle, const ::vector<T,3>& axis);

    template<typename T> friend ::vector<T,3> convertToEulerAngles(const quaternion<T>& quat);
    template<typename T> friend quaternion<T> convertToAnglesAndAxis(const quaternion<T>& quat);

    template<typename T> friend quaternion<T> slerp(const quaternion<T>& quat1, const quaternion<T>& quat2, const T& t);
};


template<typename type>
quaternion<type>::quaternion():
    s(static_cast<type>(0)),
    x(static_cast<type>(0)),
    y(static_cast<type>(0)),
    z(static_cast<type>(0))
{}

template<typename type>
quaternion<type>::quaternion(const quaternion<type>& other):
    s(other.s),
    x(other.x),
    y(other.y),
    z(other.z)
{}

template<typename type>
quaternion<type>::quaternion(const type& s,const type& x,const type& y,const type& z):
    s(s),
    x(x),
    y(y),
    z(z)
{}

template<typename type>
quaternion<type>::quaternion(const type& s, const ::vector<type, 3>& v):
    s(s),
    x(static_cast<type>(v[0])),
    y(static_cast<type>(v[1])),
    z(static_cast<type>(v[2]))
{}

template<typename type>
quaternion<type>& quaternion<type>::operator=(const quaternion<type>& other)
{
    s = other.s;
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
}

template<typename type>
quaternion<type>::~quaternion()
{}

template<typename type>
type                quaternion<type>::scalar()const
{
    return s;
}

template<typename type>
vector<type, 3>    quaternion<type>::vector()const
{
    return ::vector<type, 3>(x,y,z);
}

template<typename type>
bool                quaternion<type>::operator==(const quaternion<type>& other)const
{
    return x==other.x&&y==other.y&&z==other.z&&s==other.s;
}

template<typename type>
bool                quaternion<type>::operator!=(const quaternion<type>& other)const
{
    return !(x==other.x&&y==other.y&&z==other.z&&s==other.s);
}

template<typename type>
quaternion<type>    quaternion<type>::operator+(const quaternion<type>& other)const
{
    return quaternion<type>(s+other.s,x+other.x,y+other.y,z+other.z);
}

template<typename type>
quaternion<type>    quaternion<type>::operator-(const quaternion<type>& other)const
{
    return quaternion<type>(s-other.s,x-other.x,y-other.y,z-other.z);
}

template<typename type>
quaternion<type>    quaternion<type>::operator*(const quaternion<type>& other)const
{
    return quaternion<type>(
        s*other.s - (x*other.x + y*other.y + z*other.z),
        s*other.x + other.s*x + (y*other.z-z*other.y),
        s*other.y + other.s*y + (z*other.x-x*other.z),
        s*other.z + other.s*z + (x*other.y-y*other.x)
    );
}

template<typename type>
quaternion<type>&   quaternion<type>::operator+=(const quaternion<type>& other)
{
    s += other.s;
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

template<typename type>
quaternion<type>&   quaternion<type>::operator-=(const quaternion<type>& other)
{
    s -= other.s;
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
}

template<typename type>
quaternion<type>&   quaternion<type>::operator*=(const quaternion<type>& other)
{
    quaternion<type> copy(*this);
    *this = copy*other;

    return *this;
}

template<typename T>
std::ostream& operator<< (std::ostream & out, const quaternion<T>& quat)
{
    out<<quat.s<<'\t'<<quat.x<<'\t'<<quat.y<<'\t'<<quat.z;
    return out;
}

template<typename T>
quaternion<T> operator* (const T& c,const quaternion<T>& quat)
{
    return quaternion<T>(c*quat.s,c*quat.x,c*quat.y,c*quat.z);
}

template<typename type>
quaternion<type>&   quaternion<type>::normalize()
{
    type norma = s*s+x*x+y*y+z*z;
    norma = std::sqrt(norma);
    s /= norma;
    x /= norma;
    y /= norma;
    z /= norma;
    return *this;
}

template<typename type>
quaternion<type>&   quaternion<type>::conjugate()
{
    x = -x;
    y = -y;
    z = -z;
    return *this;
}

template<typename type>
quaternion<type>&   quaternion<type>::invert()
{
    quaternion<type> quat(*this);
    quaternion<type> ivNorma = quat*this->conjugate();
    ivNorma.s = std::sqrt(ivNorma.s);
    *this = ivNorma*(*this);
    return *this;
}


template<typename T>
quaternion<T>   normalize(const quaternion<T>& quat)
{
    T norma = quat.s*quat.s+quat.x*quat.x+quat.y*quat.y+quat.z*quat.z;
    norma = std::sqrt(norma);
    return quaternion<T>(quat.s/norma,quat.x/norma,quat.y/norma,quat.z/norma);
}

template<typename T>
quaternion<T>   conjugate(const quaternion<T>& quat)
{
    return quaternion<T>(quat.s,-quat.x,-quat.y,-quat.z);
}

template<typename T>
quaternion<T>   invert(const quaternion<T>& quat)
{
    quaternion<T> ivNorma = quat*conjugate(quat);
    ivNorma.s = std::sqrt(ivNorma.s);
    return ivNorma*conjugate(quat);
}

template<typename T>
quaternion<T> convert(const matrix<T,3,3>& O3)
{
    quaternion<T> quat;

    quat.s = std::sqrt(1.0f+O3[0][0]+O3[1][1]+O3[2][2])/2.0f;

    quat.z = (O3[1][0]-O3[0][1])/(T(4)*quat.s);
    quat.y = (O3[0][2]-O3[2][0])/(T(4)*quat.s);
    quat.x = (O3[2][1]-O3[1][2])/(T(4)*quat.s);

    return quat;
}

template<typename T>
matrix<T,3,3> convert(const quaternion<T>& quat)
{
    matrix<T,3,3> R;

    R[0][0] = T(1) - T(2)*(quat.y*quat.y + quat.z*quat.z);      R[0][1] = T(2)*(quat.x*quat.y - quat.z*quat.s);         R[0][2] = T(2)*(quat.x*quat.z + quat.y*quat.s);
    R[1][0] = T(2)*(quat.x*quat.y + quat.z*quat.s);             R[1][1] = T(1) - T(2)*(quat.x*quat.x + quat.z*quat.z);  R[1][2] = T(2)*(quat.y*quat.z - quat.x*quat.s);
    R[2][0] = T(2)*(quat.x*quat.z - quat.y*quat.s);             R[2][1] = T(2)*(quat.y*quat.z + quat.x*quat.s);         R[2][2] = T(1) - T(2)*(quat.x*quat.x + quat.y*quat.y);

    return R;
}

template<typename T>
matrix<T,4,4> convert4x4(const quaternion<T>& quat)
{
    matrix<T,4,4> R{0.0f};

    R[0][0] = T(1) - T(2)*(quat.y*quat.y + quat.z*quat.z);      R[0][1] = T(2)*(quat.x*quat.y - quat.z*quat.s);         R[0][2] = T(2)*(quat.x*quat.z + quat.y*quat.s);
    R[1][0] = T(2)*(quat.x*quat.y + quat.z*quat.s);             R[1][1] = T(1) - T(2)*(quat.x*quat.x + quat.z*quat.z);  R[1][2] = T(2)*(quat.y*quat.z - quat.x*quat.s);
    R[2][0] = T(2)*(quat.x*quat.z - quat.y*quat.s);             R[2][1] = T(2)*(quat.y*quat.z + quat.x*quat.s);         R[2][2] = T(1) - T(2)*(quat.x*quat.x + quat.y*quat.y);
    R[3][3] = T(1);

    return R;
}

template<typename T>
quaternion<T> convert(const T& yaw, const T& pitch, const T& roll)
{
    T cosy = std::cos(yaw*T(0.5));
    T siny = std::sin(yaw*T(0.5));
    T cosp = std::cos(pitch*T(0.5));
    T sinp = std::sin(pitch*T(0.5));
    T cosr = std::cos(roll*T(0.5));
    T sinr = std::sin(roll*T(0.5));

    T s = cosy*cosp*cosr + siny*sinp*sinr;
    T x = sinr*cosp*cosy - cosr*sinp*siny;
    T y = cosr*sinp*cosy + sinr*cosp*siny;
    T z = cosr*cosp*siny - sinr*sinp*cosy;

    return quaternion<T>(s,x,y,z);
}

template<typename T>
quaternion<T> convert(const T& angle, const vector<T,3>& axis)
{
    return quaternion<T>(std::cos(angle*T(0.5)),std::sin(angle*T(0.5))*vector<T,3>(axis[0],axis[1],axis[2]));
}

template<typename T>
vector<T,3> convertToEulerAngles(const quaternion<T>& quat)
{
    return  vector<T,3>(std::atan((quat.s*quat.x+quat.y*quat.z)*T(2)/(T(1)-(quat.x*quat.x+quat.y*quat.y)*T(2))),
                            std::asin((quat.s*quat.y-quat.x*quat.z)*T(2)),
                            std::atan((quat.s*quat.z+quat.y*quat.x)*T(2)/(T(1)-(quat.z*quat.z+quat.y*quat.y)*T(2))));
}

template<typename T>
quaternion<T> convertToAnglesAndAxis(const quaternion<T>& quat)
{
    return quaternion<T>(   std::acos(quat.s)*T(2),
                            vector<T,3>(quat.x,quat.y,quat.z)/std::sqrt(T(1)-quat.s*quat.s));
}

template<typename T>
quaternion<T> slerp(const quaternion<T>& quat1, const quaternion<T>& quat2, const T& t)
{
    T q1q2 = quat1.s*quat2.s + quat1.x*quat2.x + quat1.y*quat2.y + quat1.z*quat2.z;
    T modq1 = std::sqrt(quat1.s*quat1.s + quat1.x*quat1.x + quat1.y*quat1.y + quat1.z*quat1.z);
    T modq2 = std::sqrt(quat2.s*quat2.s + quat2.x*quat2.x + quat2.y*quat2.y + quat2.z*quat2.z);
    T theta = modq1*modq2 == 0.0f ? 0.0f : std::acos(q1q2/modq1/modq2);
    theta = std::isnan(theta) ? 0.0f : theta;

    auto sinRatio = [](const T& theta, const T& t){
        return theta == T(0) ? t : std::sin(t*theta)/std::sin(theta);
    };

    return sinRatio(theta, T(1)-t) * quat1 + sinRatio(theta, t) * quat2;
}

template<typename type>
matrix<type,4,4> rotate(quaternion<type> qu){
    return convert4x4(qu);
}

#endif // QUATERNION_H
