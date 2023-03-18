#ifndef QUATERNION_H
#define QUATERNION_H

#include "glm/glm.hpp"
#include <iostream>

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
    quaternion(const type& s,const glm::vec<3,type,glm::defaultp>& v);
    quaternion<type>& operator=(const quaternion<type>& other);
    ~quaternion();

    type                                 scalar()const;
    glm::vec<3,type,glm::defaultp>       vector()const;

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

    template<typename T> friend quaternion<T> convert(const glm::mat<3,3,T,glm::defaultp>& O3);
    template<typename T> friend glm::mat<3,3,T,glm::defaultp> convert(const quaternion<T>& quat);

    template<typename T> friend quaternion<T> convert(const T& yaw, const T& pitch, const T& roll);
    template<typename T> friend quaternion<T> convert(const T& angle, const glm::vec<3,type,glm::defaultp>& axis);

    template<typename T> friend glm::vec<3,T,glm::defaultp> convertToEulerAngles(const quaternion<T>& quat);
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
quaternion<type>::quaternion(const type& s,const glm::vec<3,type,glm::defaultp>& v):
    s(s),
    x(static_cast<type>(v.x)),
    y(static_cast<type>(v.y)),
    z(static_cast<type>(v.z))
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
glm::vec<3,type,glm::defaultp>           quaternion<type>::vector()const
{
    return glm::vec<3,type,glm::defaultp>(x,y,z);
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
    norma = glm::sqrt(norma);
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
    ivNorma.s = glm::sqrt(ivNorma.s);
    *this = ivNorma*(*this);
    return *this;
}


template<typename T>
quaternion<T>   normalize(const quaternion<T>& quat)
{
    T norma = quat.s*quat.s+quat.x*quat.x+quat.y*quat.y+quat.z*quat.z;
    norma = glm::sqrt(norma);
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
    ivNorma.s = glm::sqrt(ivNorma.s);
    return ivNorma*conjugate(quat);
}

template<typename T>
quaternion<T> convert(const glm::mat<3,3,T,glm::defaultp>& O3)
{
    quaternion<T> quat;

    quat.s = glm::sqrt(1.0f+O3[0][0]+O3[1][1]+O3[2][2])/2.0f;

    quat.z = (O3[1][0]-O3[0][1])/(T(4)*quat.s);
    quat.y = (O3[0][2]-O3[2][0])/(T(4)*quat.s);
    quat.x = (O3[2][1]-O3[1][2])/(T(4)*quat.s);

    return quat;
}

template<typename T>
glm::mat<3,3,T,glm::defaultp> convert(const quaternion<T>& quat)
{
    glm::mat<3,3,T,glm::defaultp> R;

    R[0][0] = T(1) - T(2)*(quat.y*quat.y + quat.z*quat.z);      R[0][1] = T(2)*(quat.x*quat.y - quat.z*quat.s);         R[0][2] = T(2)*(quat.x*quat.z + quat.y*quat.s);
    R[1][0] = T(2)*(quat.x*quat.y + quat.z*quat.s);             R[1][1] = T(1) - T(2)*(quat.x*quat.x + quat.z*quat.z);  R[1][2] = T(2)*(quat.y*quat.z - quat.x*quat.s);
    R[2][0] = T(2)*(quat.x*quat.z - quat.y*quat.s);             R[2][1] = T(2)*(quat.y*quat.z + quat.x*quat.s);         R[2][2] = T(1) - T(2)*(quat.x*quat.x + quat.y*quat.y);

    return R;
}

template<typename T>
quaternion<T> convert(const T& yaw, const T& pitch, const T& roll)
{
    T cosy = glm::cos(yaw*T(0.5));
    T siny = glm::sin(yaw*T(0.5));
    T cosp = glm::cos(pitch*T(0.5));
    T sinp = glm::sin(pitch*T(0.5));
    T cosr = glm::cos(roll*T(0.5));
    T sinr = glm::sin(roll*T(0.5));

    T s = cosy*cosp*cosr + siny*sinp*sinr;
    T x = sinr*cosp*cosy - cosr*sinp*siny;
    T y = cosr*sinp*cosy + sinr*cosp*siny;
    T z = cosr*cosp*siny - sinr*sinp*cosy;

    return quaternion<T>(s,x,y,z);
}

template<typename T>
quaternion<T> convert(const T& angle, const glm::vec<3,T,glm::defaultp>& axis)
{
    return quaternion<T>(glm::cos(angle*T(0.5)),glm::sin(angle*T(0.5))*glm::vec<3,T,glm::defaultp>(axis.x,axis.y,axis.z));
}

template<typename T>
glm::vec<3,T,glm::defaultp> convertToEulerAngles(const quaternion<T>& quat)
{
    return  glm::vec<3,T,glm::defaultp>(  glm::atan((quat.s*quat.x+quat.y*quat.z)*T(2)/(T(1)-(quat.x*quat.x+quat.y*quat.y)*T(2))),
                                          glm::asin((quat.s*quat.y-quat.x*quat.z)*T(2)),
                                          glm::atan((quat.s*quat.z+quat.y*quat.x)*T(2)/(T(1)-(quat.z*quat.z+quat.y*quat.y)*T(2))));
}

template<typename T>
quaternion<T> convertToAnglesAndAxis(const quaternion<T>& quat)
{
    return quaternion<T>(   glm::acos(quat.s)*T(2),
                            glm::vec<3,T,glm::defaultp>(quat.x,quat.y,quat.z)/glm::sqrt(T(1)-quat.s*quat.s));
}

template<typename T>
quaternion<T> slerp(const quaternion<T>& quat1, const quaternion<T>& quat2, const T& t)
{
    T q1q2 = quat1.s*quat2.s + quat1.x*quat2.x + quat1.y*quat2.y + quat1.z*quat2.z;
    T modq1 = glm::sqrt(quat1.s*quat1.s + quat1.x*quat1.x + quat1.y*quat1.y + quat1.z*quat1.z);
    T modq2 = glm::sqrt(quat2.s*quat2.s + quat2.x*quat2.x + quat2.y*quat2.y + quat2.z*quat2.z);
    T theta = glm::acos(q1q2/modq1/modq2);

    return glm::sin((T(1)-t)*theta)/glm::sin(theta)*quat1 + glm::sin(t*theta)/glm::sin(theta)*quat2;
}

#endif // QUATERNION_H
