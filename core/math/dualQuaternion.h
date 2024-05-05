#ifndef DUALQUATERNION_H
#define DUALQUATERNION_H

#include "quaternion.h"

namespace moon::math {

template<typename type>
class DualQuaternion
{
private:
  Quaternion<type> p;
  Quaternion<type> q;
public:
  DualQuaternion() = default;
  DualQuaternion(const DualQuaternion<type>& other);
  DualQuaternion(const Quaternion<type>& p, const Quaternion<type>& q);
  DualQuaternion<type>& operator=(const DualQuaternion<type>& other);

  Quaternion<type>          rotation()const;
  Quaternion<type>          translation()const;

  bool                      operator==(const DualQuaternion<type>& other)const;
  bool                      operator!=(const DualQuaternion<type>& other)const;
  DualQuaternion<type>      operator+ (const DualQuaternion<type>& other)const;
  DualQuaternion<type>      operator- (const DualQuaternion<type>& other)const;
  DualQuaternion<type>      operator* (const DualQuaternion<type>& other)const;
  DualQuaternion<type>&     operator+=(const DualQuaternion<type>& other);
  DualQuaternion<type>&     operator-=(const DualQuaternion<type>& other);
  DualQuaternion<type>&     operator*=(const DualQuaternion<type>& other);

  DualQuaternion<type>&     normalize();
  DualQuaternion<type>&     conjugate();
  DualQuaternion<type>&     invert();

  template<typename T> friend DualQuaternion<T>   normalize(const DualQuaternion<T>& quat);
  template<typename T> friend DualQuaternion<T>   conjugate(const DualQuaternion<T>& quat);
  template<typename T> friend DualQuaternion<T>   invert(const DualQuaternion<T>& quat);

  template<typename T> friend DualQuaternion<T> operator* (const T& c, const DualQuaternion<T>& quat);
  template<typename T> friend std::ostream& operator<< (std::ostream & out, const DualQuaternion<T>& quat);

  template<typename T> friend DualQuaternion<T> convert(const Quaternion<T>& rotation, const Quaternion<T>& translation);

  template<typename T> friend DualQuaternion<T> convert(const Matrix<T,4,4>& SE3);
  template<typename T> friend Matrix<T,4,4> convert(const DualQuaternion<T>& quat);

  template<typename T> friend DualQuaternion<T> slerp(const DualQuaternion<T>& quat1, const DualQuaternion<T>& quat2, const T& t);
};

template<typename type>
DualQuaternion<type>::DualQuaternion(const Quaternion<type>& p, const Quaternion<type>& q):
    p(p),q(q)
{}

template<typename type>
DualQuaternion<type>::DualQuaternion(const DualQuaternion<type>& other):
    p(other.p),q(other.q)
{}

template<typename type>
DualQuaternion<type>& DualQuaternion<type>::operator=(const DualQuaternion<type>& other)
{
    p = other.p;
    q = other.q;
    return *this;
}

template<typename type>
Quaternion<type>          DualQuaternion<type>::rotation()const
{
    return p;
}

template<typename type>
Quaternion<type>          DualQuaternion<type>::translation()const
{
    Quaternion<type> copy(p);
    copy.conjugate();
    return type(2)*q*copy;
}

template<typename type>
bool                      DualQuaternion<type>::operator==(const DualQuaternion<type>& other)const
{
    return p==other.p&&q==other.q;
}

template<typename type>
bool                      DualQuaternion<type>::operator!=(const DualQuaternion<type>& other)const
{
    return !(p==other.p&&q==other.q);
}

template<typename type>
DualQuaternion<type>      DualQuaternion<type>::operator+ (const DualQuaternion<type>& other)const
{
    return DualQuaternion<type>(p+other.p,q+other.q);
}

template<typename type>
DualQuaternion<type>      DualQuaternion<type>::operator- (const DualQuaternion<type>& other)const
{
    return DualQuaternion<type>(p-other.p,q-other.q);
}

template<typename type>
DualQuaternion<type>      DualQuaternion<type>::operator* (const DualQuaternion<type>& other)const
{
    return DualQuaternion<type>(
        p*other.p,
        p*other.q+q*other.p
    );
}

template<typename type>
DualQuaternion<type>&     DualQuaternion<type>::operator+=(const DualQuaternion<type>& other)
{
    p += other.p;
    q += other.q;
    return *this;
}

template<typename type>
DualQuaternion<type>&     DualQuaternion<type>::operator-=(const DualQuaternion<type>& other)
{
    p -= other.p;
    q -= other.q;
    return *this;
}

template<typename type>
DualQuaternion<type>&     DualQuaternion<type>::operator*=(const DualQuaternion<type>& other)
{
    DualQuaternion<type> copy(*this);
    *this = copy*other;

    return *this;
}

template<typename T>
std::ostream& operator<< (std::ostream & out, const DualQuaternion<T>& quat)
{
    out<<quat.p<<"\t\t"<<quat.q;
    return out;
}

template<typename T>
DualQuaternion<T> operator* (const T& c,const DualQuaternion<T>& quat)
{
    return DualQuaternion<T>(c*quat.p,c*quat.q);
}

template<typename type>
DualQuaternion<type>&   DualQuaternion<type>::normalize()
{
    if(type norma = p.re()*p.re()+dot(p.im(),p.im()); norma != 0){
        norma = type(1)/std::sqrt(norma);
        p = norma * p;
        q = norma * q;
    }
    return *this;
}

template<typename type>
DualQuaternion<type>&   DualQuaternion<type>::conjugate()
{
    p.conjugate();
    type(-1)*q.conjugate();
    return *this;
}

template<typename type>
DualQuaternion<type>&   DualQuaternion<type>::invert()
{
    p.invert();
    q = type(-1)*p*q*p;

    return *this;
}

template<typename T>
DualQuaternion<T>   normalize(const DualQuaternion<T>& quat)
{
    const DualQuaternion<T>& res(quat);
    if(quat.p.s*quat.q.s+quat.p.x*quat.q.x+quat.p.y*quat.q.y+quat.p.z*quat.q.z==0){
        T norma = quat.p.s*quat.p.s+quat.p.x*quat.p.x+quat.p.y*quat.p.y+quat.p.z*quat.p.z;
        norma = std::sqrt(norma);

        res.p.s = quat.p.s / norma;
        res.p.x = quat.p.x / norma;
        res.p.y = quat.p.y / norma;
        res.p.z = quat.p.z / norma;
        res.q.s = quat.q.s / norma;
        res.q.x = quat.q.x / norma;
        res.q.y = quat.q.y / norma;
        res.q.z = quat.q.z / norma;

    }
    return res;
}

template<typename T>
DualQuaternion<T>   conjugate(const DualQuaternion<T>& quat)
{
    return DualQuaternion<T>(conjugate(quat.p),T(-1)*conjugate(quat.q));
}

template<typename T>
DualQuaternion<T>   invert(const DualQuaternion<T>& quat)
{
    DualQuaternion<T> res(quat);
    res.p.invert();
    res.q = T(-1)*res.p*res.q*res.p;

    return res;
}

template<typename T>
DualQuaternion<T>   convert(const Quaternion<T>& rotation, const Quaternion<T>& translation)
{
    return DualQuaternion<T>(rotation,T(0.5)*translation*rotation);
}

template<typename T>
Matrix<T,4,4> convert(const DualQuaternion<T>& quat)
{
    Quaternion<T> rotatrion = quat.rotation();
    Quaternion<T> translation = quat.translation();

    Matrix<T,3,3> R = convert(rotatrion);

    Matrix<T,4,4> SE3;

    SE3[0][0] = R[0][0];    SE3[0][1] = R[0][1];    SE3[0][2] = R[0][2];    SE3[0][3] = translation.im()[0];
    SE3[1][0] = R[1][0];    SE3[1][1] = R[1][1];    SE3[1][2] = R[1][2];    SE3[1][3] = translation.im()[1];
    SE3[2][0] = R[2][0];    SE3[2][1] = R[2][1];    SE3[2][2] = R[2][2];    SE3[2][3] = translation.im()[2];
    SE3[3][0] = T(0);       SE3[3][1] = T(0);       SE3[3][2] = T(0);       SE3[3][3] = T(1);

    return SE3;
}

template<typename T>
DualQuaternion<T> convert(const Matrix<T,4,4>& SE3)
{
    Matrix<T,3,3> R;

    R[0][0] = SE3[0][0];    R[0][1] = SE3[0][1];    R[0][2] = SE3[0][2];
    R[1][0] = SE3[1][0];    R[1][1] = SE3[1][1];    R[1][2] = SE3[1][2];
    R[2][0] = SE3[2][0];    R[2][1] = SE3[2][1];    R[2][2] = SE3[2][2];

    Quaternion<T> rotatrion = convert(R);
    Quaternion<T> translation = Quaternion<T>(T(0),SE3[0][3],SE3[1][3],SE3[2][3]);

    return convert(rotatrion,translation);
}


template<typename T>
DualQuaternion<T> slerp(const DualQuaternion<T>& quat1, const DualQuaternion<T>& quat2, const T& t)
{
    Quaternion<T> r1 = quat1.rotation();
    Quaternion<T> r2 = quat2.rotation();
    Quaternion<T> t1 = quat1.translation();
    Quaternion<T> t2 = quat2.translation();

    DualQuaternion<T> dQuat(conjugate(r1) * r2, T(0.5) * conjugate(r1) * (t2 - t1) * r2);
    dQuat.normalize();

    Vector<T,3> l = normalize(dQuat.p.im());
    Vector<T,3> tr = T(0.5) * Vector<T,3>(dQuat.translation().im());

    T d = dot(tr,l);
    T theta = std::acos(dQuat.p.re());

    auto ratio = [](const T& theta, const T& t){
        return theta == T(0) ? t : std::sin(t * theta)/std::tan(theta);
    };

    DualQuaternion<T> sigma(
        Quaternion<T>(std::cos(t * theta), std::sin(t * theta) * l),
        Quaternion<T>(std::sin(t * theta) * (- d*t), std::cos(t * theta) * (l * d*t) + std::sin(t * theta) * cross(tr,l) + ratio(theta,t) * (tr - d*l))
    );

    return quat1*sigma;
}

extern template class DualQuaternion<float>;
extern template class DualQuaternion<double>;

}
#endif // DUALQUATERNION_H
