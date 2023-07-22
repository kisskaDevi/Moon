#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include <math.h>

template<typename type, uint32_t n>
class baseVector
{
protected:
    baseVector<type, n - 1> vec;
    type s{type(0)};

public:
    baseVector() = default;
    baseVector(const baseVector<type, n - 1>& other);
    baseVector(const baseVector<type, n>& other);
    baseVector(const baseVector<type, n - 1>& other, const type& s);
    baseVector<type, n>& operator=(const baseVector<type, n>& other);

    type& operator[](uint32_t i);
    const type& operator[](uint32_t i) const;
    uint32_t size() const;

    bool operator==(const baseVector<type, n>& other) const;
    bool operator!=(const baseVector<type, n>& other) const;

    baseVector<type,n> operator+(const baseVector<type, n>& other) const;
    baseVector<type,n> operator-(const baseVector<type, n>& other) const;
    baseVector<type,n> operator*(const baseVector<type, n>& other) const;
    baseVector<type,n> operator/(const type& c) const;

    baseVector<type,n>& operator+=(const baseVector<type, n>& other);
    baseVector<type,n>& operator-=(const baseVector<type, n>& other);
    baseVector<type,n>& operator*=(const baseVector<type, n>& other);

    baseVector<type,n>& operator+=(const type& c);
    baseVector<type,n>& operator-=(const type& c);
    baseVector<type,n>& operator*=(const type& c);
    baseVector<type,n>& operator/=(const type& c);

    template<typename T, uint32_t N> friend baseVector<T,N> operator+(const T& c, const baseVector<T,N>& other);
    template<typename T, uint32_t N> friend baseVector<T,N> operator-(const T& c, const baseVector<T,N>& other);
    template<typename T, uint32_t N> friend baseVector<T,N> operator*(const T& c, const baseVector<T,N>& other);

    template<typename T, uint32_t N> friend baseVector<T,N> operator+(const baseVector<T,N>& other, const T& c);
    template<typename T, uint32_t N> friend baseVector<T,N> operator-(const baseVector<T,N>& other, const T& c);
    template<typename T, uint32_t N> friend baseVector<T,N> operator*(const baseVector<T,N>& other, const T& c);

    baseVector<type,n>& normalize();
    template<typename T, uint32_t N> friend baseVector<T,N> normalize(const baseVector<T,N>& other);

    template<typename T, uint32_t N> friend T dot(const baseVector<T,N>& left, const baseVector<T,N>& right);

    template<typename T, uint32_t N> friend std::ostream& operator<<(std::ostream& out, const baseVector<T,N>& other);
};

template<typename type, uint32_t n>
baseVector<type,n>::baseVector(const baseVector<type, n - 1>& other)
    : vec(other) {}

template<typename type, uint32_t n>
baseVector<type,n>::baseVector(const baseVector<type, n>& other)
    : vec(other.vec), s(other.s) {}

template<typename type, uint32_t n>
baseVector<type,n>::baseVector(const baseVector<type, n - 1>& other, const type& s)
    : vec(other), s(s) {}

template<typename type, uint32_t n>
baseVector<type, n>& baseVector<type,n>::operator=(const baseVector<type,n>& other){
    vec = other.vec;
    s = other.s;
    return *this;
}

template<typename type, uint32_t n>
type& baseVector<type,n>::operator[](uint32_t i) {
    return i >= n - 1 ? s : vec[i];
}

template<typename type, uint32_t n>
const type& baseVector<type,n>::operator[](uint32_t i) const{
    return i >= n - 1 ? s : vec[i];
}

template<typename type, uint32_t n>
uint32_t baseVector<type,n>::size() const {
    return vec.size() + 1;
}

template<typename type, uint32_t n>
bool baseVector<type,n>::operator==(const baseVector<type,n>& other) const {
    return vec == other.vec && s == other.s;
}

template<typename type, uint32_t n>
bool baseVector<type,n>::operator!=(const baseVector<type,n>& other) const {
    return !(*this == other);
}

template<typename type, uint32_t n>
baseVector<type,n> baseVector<type,n>::operator+(const baseVector<type,n>& other) const {
    return baseVector<type,n>(vec + other.vec, s + other.s);
}

template<typename type, uint32_t n>
baseVector<type,n> baseVector<type,n>::operator-(const baseVector<type,n>& other) const {
    return baseVector<type,n>(vec - other.vec, s - other.s);
}

template<typename type, uint32_t n>
baseVector<type,n> baseVector<type,n>::operator*(const baseVector<type,n>& other) const {
    return baseVector<type,n>(vec * other.vec, s * other.s);
}

template<typename type, uint32_t n>
baseVector<type,n> baseVector<type,n>::operator/(const type& c) const {
    return baseVector<type,n>(vec / c, s / c);
}

template<typename type, uint32_t n>
baseVector<type,n>& baseVector<type,n>::operator+=(const baseVector<type,n>& other) {
    vec += other.vec; s += other.s;
    return *this;
}

template<typename type, uint32_t n>
baseVector<type,n>& baseVector<type,n>::operator-=(const baseVector<type,n>& other) {
    vec -= other.vec; s -= other.s;
    return *this;
}

template<typename type, uint32_t n>
baseVector<type,n>& baseVector<type,n>::operator*=(const baseVector<type,n>& other) {
    vec *= other.vec; s *= other.s;
    return *this;
}

template<typename type, uint32_t n>
baseVector<type,n>& baseVector<type,n>::operator+=(const type& c) {
    vec += c; s += c;
    return *this;
}

template<typename type, uint32_t n>
baseVector<type,n>& baseVector<type,n>::operator-=(const type& c) {
    vec -= c; s -= c;
    return *this;
}

template<typename type, uint32_t n>
baseVector<type,n>& baseVector<type,n>::operator*=(const type& c) {
    vec *= c; s *= c;
    return *this;
}

template<typename type, uint32_t n>
baseVector<type,n>& baseVector<type,n>::operator/=(const type& c) {
    vec /= c; s /= c;
    return *this;
}

template<typename T, uint32_t N> baseVector<T,N> operator+(const T& c, const baseVector<T, N>& other) {
    return baseVector<T,N>(c + other.vec, c + other.s);
}

template<typename T, uint32_t N> baseVector<T,N> operator-(const T& c, const baseVector<T, N>& other) {
    return baseVector<T,N>(c - other.vec, c - other.s);
}

template<typename T, uint32_t N> baseVector<T,N> operator*(const T& c, const baseVector<T, N>& other) {
    return baseVector<T,N>(c * other.vec, c * other.s);
}

template<typename T, uint32_t N> baseVector<T,N> operator+(const baseVector<T, N>& other, const T& c) {
    return baseVector<T,N>(other.vec + c, other.s + c);
}

template<typename T, uint32_t N> baseVector<T,N> operator-(const baseVector<T, N>& other, const T& c) {
    return baseVector<T,N>(other.vec - c, other.s - c);
}

template<typename T, uint32_t N> baseVector<T,N> operator*(const baseVector<T, N>& other, const T& c) {
    return baseVector<T,N>(other.vec * c, other.s * c);
}

template<typename T, uint32_t N> T dot(const baseVector<T, N>& left, const baseVector<T, N>& right){
    return dot(left.vec, right.vec) + left.s * right.s;
}

template<typename type, uint32_t n>
baseVector<type,n>& baseVector<type,n>::normalize(){
    type norma = type(1) / std::sqrt(dot(*this, *this));
    return *this *= norma;
}

template<typename T, uint32_t N> baseVector<T,N> normalize(const baseVector<T, N>& other) {
    T norma = T(1) / std::sqrt(dot(other, other));
    return other * norma;
}

template<typename T, uint32_t N> std::ostream& operator<<(std::ostream& out, const baseVector<T, N>& other){
    out << other.vec << '\t' << other.s;
    return out;
}

template<typename type>
class baseVector<type, 2>
{
protected:
    type x0{type(0)}, x1{type(0)};

public:
    baseVector() = default;
    baseVector(const baseVector<type, 2>& other);
    baseVector<type, 2>& operator=(const baseVector<type, 2>& other);
    baseVector(const type& x0, const type& x1);

    type& operator[](uint32_t i);
    const type& operator[](uint32_t i) const;
    uint32_t size() const;

    bool operator==(const baseVector<type, 2>& other) const;
    bool operator!=(const baseVector<type, 2>& other) const;

    baseVector<type,2> operator+(const baseVector<type, 2>& other) const;
    baseVector<type,2> operator-(const baseVector<type, 2>& other) const;
    baseVector<type,2> operator*(const baseVector<type, 2>& other) const;
    baseVector<type,2> operator/(const type& c) const;

    baseVector<type,2>& operator+=(const baseVector<type, 2>& other);
    baseVector<type,2>& operator-=(const baseVector<type, 2>& other);
    baseVector<type,2>& operator*=(const baseVector<type, 2>& other);

    baseVector<type,2>& operator+=(const type& c);
    baseVector<type,2>& operator-=(const type& c);
    baseVector<type,2>& operator*=(const type& c);
    baseVector<type,2>& operator/=(const type& c);

    template<typename T> friend baseVector<T,2> operator+(const T& c, const baseVector<T, 2>& other);
    template<typename T> friend baseVector<T,2> operator-(const T& c, const baseVector<T, 2>& other);
    template<typename T> friend baseVector<T,2> operator*(const T& c, const baseVector<T, 2>& other);

    template<typename T> friend baseVector<T,2> operator+(const baseVector<T, 2>& other, const T& c);
    template<typename T> friend baseVector<T,2> operator-(const baseVector<T, 2>& other, const T& c);
    template<typename T> friend baseVector<T,2> operator*(const baseVector<T, 2>& other, const T& c);

    baseVector<type,2>& normalize();
    template<typename T> friend baseVector<T,2> normalize(const baseVector<T, 2>& other);

    template<typename T> friend T dot(const baseVector<T, 2>& left, const baseVector<T, 2>& right);

    template<typename T> friend std::ostream& operator<<(std::ostream& out, const baseVector<T, 2>& other);
};

template<typename type>
baseVector<type,2>::baseVector(const baseVector<type, 2>& other)
    : x0(other.x0), x1(other.x1) {}

template<typename type>
baseVector<type, 2>& baseVector<type,2>::operator=(const baseVector<type, 2>& other){
    x0 = other.x0;
    x1 = other.x1;
    return *this;
}

template<typename type>
baseVector<type,2>::baseVector(const type& x0, const type& x1)
    : x0(x0), x1(x1) {}

template<typename type>
type& baseVector<type,2>::operator[](uint32_t i) {
    return i >= 1 ? x1 : x0;
}

template<typename type>
const type& baseVector<type,2>::operator[](uint32_t i) const {
    return i >= 1 ? x1 : x0;
}

template<typename type>
uint32_t baseVector<type,2>::size() const {
    return 2;
}

template<typename type>
bool baseVector<type,2>::operator==(const baseVector<type, 2>& other) const {
    return x0 == other.x0 && x1 == other.x1;
}

template<typename type>
bool baseVector<type,2>::operator!=(const baseVector<type,2>& other) const {
    return !(*this == other);
}

template<typename type>
baseVector<type,2> baseVector<type,2>::operator+(const baseVector<type,2>& other) const {
    return baseVector<type,2>(x0 + other.x0, x1 + other.x1);
}

template<typename type>
baseVector<type,2> baseVector<type,2>::operator-(const baseVector<type,2>& other) const {
    return baseVector<type,2>(x0 - other.x0, x1 - other.x1);
}

template<typename type>
baseVector<type,2> baseVector<type,2>::operator*(const baseVector<type,2>& other) const {
    return baseVector<type,2>(x0 * other.x0, x1 * other.x1);
}

template<typename type>
baseVector<type,2> baseVector<type,2>::operator/(const type& c) const {
    return baseVector<type,2>(x0 / c, x1 / c);
}

template<typename type>
baseVector<type,2>& baseVector<type,2>::operator+=(const baseVector<type,2>& other) {
    x0 += other.x0; x1 += other.x1;
    return *this;
}

template<typename type>
baseVector<type,2>& baseVector<type,2>::operator-=(const baseVector<type,2>& other) {
    x0 -= other.x0; x1 -= other.x1;
    return *this;
}

template<typename type>
baseVector<type,2>& baseVector<type,2>::operator*=(const baseVector<type,2>& other) {
    x0 *= other.x0; x1 *= other.x1;
    return *this;
}

template<typename type>
baseVector<type,2>& baseVector<type,2>::operator+=(const type& c) {
    x0 += c; x1 += c;
    return *this;
}

template<typename type>
baseVector<type,2>& baseVector<type,2>::operator-=(const type& c) {
    x0 -= c; x1 -= c;
    return *this;
}

template<typename type>
baseVector<type,2>& baseVector<type,2>::operator*=(const type& c) {
    x0 *= c; x1 *= c;
    return *this;
}

template<typename type>
baseVector<type,2>& baseVector<type,2>::operator/=(const type& c) {
    x0 /= c; x1 /= c;
    return *this;
}

template<typename T> baseVector<T,2> operator+(const T& c, const baseVector<T,2>& other) {
    return baseVector<T,2>(c + other.x0, c + other.x1);
}

template<typename T> baseVector<T,2> operator-(const T& c, const baseVector<T,2>& other) {
    return baseVector<T,2>(c - other.x0, c - other.x1);
}

template<typename T> baseVector<T,2> operator*(const T& c, const baseVector<T,2>& other) {
    return baseVector<T,2>(c * other.x0, c * other.x1);
}

template<typename T> baseVector<T,2> operator+(const baseVector<T,2>& other, const T& c) {
    return baseVector<T,2>(other.x0 + c, other.x1 + c);
}

template<typename T> baseVector<T,2> operator-(const baseVector<T,2>& other, const T& c) {
    return baseVector<T,2>(other.x0 - c, other.x1 - c);
}

template<typename T> baseVector<T,2> operator*(const baseVector<T,2>& other, const T& c) {
    return baseVector<T,2>(other.x0 * c, other.x1 * c);
}

template<typename type>
baseVector<type,2>& baseVector<type,2>::normalize(){
    type n = type(1) / std::sqrt(x0 * x0 + x1 * x1);
    return *this *= n;
}

template<typename T> baseVector<T,2> normalize(const baseVector<T,2>& other) {
    T n = T(1) / std::sqrt(other.x0 * other.x0 + other.x1 * other.x1);
    return other * n;
}

template<typename T> T dot(const baseVector<T,2>& left, const baseVector<T,2>& right){
    return left.x0 * right.x0 + left.x1 * right.x1;
}

template<typename T> std::ostream& operator<<(std::ostream& out, const baseVector<T,2>& other) {
    out << other.x0 << '\t' << other.x1;
    return out;
}

template<typename type, uint32_t n> class vector;

template<typename type>
class vector<type, 2> : public baseVector<type, 2>
{
public:
    vector() : baseVector<type,2>() {}
    vector(const type& x0, const type& x1) {
        this->x0 = x0;
        this->x1 = x1;
    }
};

template<typename type>
class vector<type, 3> : public baseVector<type, 3>
{
public:
    vector() : baseVector<type,3>() {}
    vector(const type& x0, const type& x1, const type& s) {
        this->vec = baseVector<type, 2>(x0, x1);
        this->s = s;
    }

    template<typename T> friend vector<T,3> cross(const vector<T, 3>& left, const vector<T, 3>& right);
};

template<typename type>
class vector<type, 4> : public baseVector<type, 4>
{
public:
    vector() : baseVector<type,4>() {}
    vector(const type& x0, const type& x1, const type& x2, const type& s) {
        this->vec = baseVector<type, 3>({x0, x1}, x2);
        this->s = s;
    }
};

template<typename T> vector<T,3> cross(const vector<T, 3>& left, const vector<T, 3>& right){
    return vector<T,3>(
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0]
    );
}

#endif // VECTOR_H
