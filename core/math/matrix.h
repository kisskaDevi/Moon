#ifndef MATRIX_H
#define MATRIX_H

#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <math.h>

#include "vector.h"

template<typename type, uint32_t n, uint32_t m>
class baseMatrix {
protected:
    baseVector<type, m> mat[n];

public:
    baseMatrix() = default;
    baseMatrix(const baseMatrix<type,n,m>& other);
    baseMatrix<type,n,m>& operator=(const baseMatrix<type,n,m>& other);

    baseVector<type, m>& operator[](uint32_t i);
    const baseVector<type, m>& operator[](uint32_t i) const;

    bool operator==(const baseMatrix<type, n, m>& other) const;
    bool operator!=(const baseMatrix<type, n, m>& other) const;

    baseMatrix<type, n, m> operator+(const baseMatrix<type, n, m>& other) const;
    baseMatrix<type, n, m> operator-(const baseMatrix<type, n, m>& other) const;
    baseVector<type, n> operator*(const baseVector<type, m>& other) const;
    baseMatrix<type, n, m> operator/(const type& c) const;

    baseMatrix<type, n, m>& operator+=(const baseMatrix<type, n, m>& other);
    baseMatrix<type, n, m>& operator-=(const baseMatrix<type, n, m>& other);

    baseMatrix<type, n, m>& operator+=(const type& c);
    baseMatrix<type, n, m>& operator-=(const type& c);
    baseMatrix<type, n, m>& operator*=(const type& c);
    baseMatrix<type, n, m>& operator/=(const type& c);

    baseMatrix<type, n-1, m-1> extract(uint32_t i, uint32_t j) const;

    template<typename T, uint32_t M, uint32_t N> friend baseMatrix<T,N,M> transpose(const baseMatrix<T,N,M>& other);

    template<typename T, uint32_t N, uint32_t M> friend baseMatrix<T,N,M> operator+(const T& c, const baseMatrix<T,N,M>& other);
    template<typename T, uint32_t N, uint32_t M> friend baseMatrix<T,N,M> operator-(const T& c, const baseMatrix<T,N,M>& other);
    template<typename T, uint32_t N, uint32_t M> friend baseMatrix<T,N,M> operator*(const T& c, const baseMatrix<T,N,M>& other);

    template<typename T, uint32_t N, uint32_t M> friend baseMatrix<T,N,M> operator+(const baseMatrix<T,N,M>& other, const T& c);
    template<typename T, uint32_t N, uint32_t M> friend baseMatrix<T,N,M> operator-(const baseMatrix<T,N,M>& other, const T& c);
    template<typename T, uint32_t N, uint32_t M> friend baseMatrix<T,N,M> operator*(const baseMatrix<T,N,M>& other, const T& c);

    template<typename T, uint32_t N, uint32_t M, uint32_t K> friend baseMatrix<T,N,K> operator*(const baseMatrix<T,N,M>& left, const baseMatrix<T,M,K>& right);

    template<typename T, uint32_t N, uint32_t M> friend std::ostream& operator<<(std::ostream& out, const baseMatrix<T,N,M>& other);
};

template<typename type, uint32_t n, uint32_t m>
baseMatrix<type,n,m>::baseMatrix(const baseMatrix<type,n,m>& other) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] = other.mat[i];
    }
}

template<typename type, uint32_t n, uint32_t m>
baseMatrix<type,n,m>& baseMatrix<type,n,m>::operator=(const baseMatrix<type,n,m>& other) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] = other.mat[i];
    }
    return *this;
}

template<typename type, uint32_t n, uint32_t m>
baseVector<type, m>& baseMatrix<type,n,m>::operator[](uint32_t i) {
    return mat[i];
}

template<typename type, uint32_t n, uint32_t m>
const baseVector<type, m>& baseMatrix<type,n,m>::operator[](uint32_t i) const {
    return mat[i];
}

template<typename type, uint32_t n, uint32_t m>
bool baseMatrix<type,n,m>::operator==(const baseMatrix<type, n, m>& other) const {
    bool result = true;
    for(uint32_t i = 0; i < n; i++) {
        result &= (mat[i] == other.mat[i]);
    }
    return result;
}

template<typename type, uint32_t n, uint32_t m>
bool baseMatrix<type,n,m>::operator!=(const baseMatrix<type, n, m>& other) const {
    return !(*this == other);
}

template<typename type, uint32_t n, uint32_t m>
baseMatrix<type, n, m> baseMatrix<type,n,m>::operator+(const baseMatrix<type, n, m>& other) const {
    baseMatrix<type,n,m> result(*this);
    for(uint32_t i = 0; i < n; i++) {
        result[i] += other.mat[i];
    }
    return result;
}

template<typename type, uint32_t n, uint32_t m>
baseMatrix<type, n, m> baseMatrix<type,n,m>::operator-(const baseMatrix<type, n, m>& other) const {
    baseMatrix<type,n,m> result(*this);
    for(uint32_t i = 0; i < n; i++) {
        result[i] -= other.mat[i];
    }
    return result;
}

template<typename type, uint32_t n, uint32_t m>
baseVector<type, n> baseMatrix<type,n,m>::operator*(const baseVector<type, m>& other) const {
    baseVector<type,n> result;
    for(uint32_t i = 0; i < n; i++) {
        result[i] = dot(mat[i], other);
    }
    return result;
}

template<typename type, uint32_t n, uint32_t m>
baseMatrix<type, n, m> baseMatrix<type,n,m>::operator/(const type& c) const {
    baseMatrix<type,n,m> result(*this);
    for(uint32_t i = 0; i < n; i++) {
        result[i] /= c;
    }
    return result;
}

template<typename type, uint32_t n, uint32_t m>
baseMatrix<type, n, m>& baseMatrix<type,n,m>::operator+=(const baseMatrix<type, n, m>& other) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] += other.mat[i];
    }
    return *this;
}

template<typename type, uint32_t n, uint32_t m>
baseMatrix<type, n, m>& baseMatrix<type,n,m>::operator-=(const baseMatrix<type, n, m>& other) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] -= other.mat[i];
    }
    return *this;
}

template<typename type, uint32_t n, uint32_t m>
baseMatrix<type, n, m>& baseMatrix<type,n,m>::operator+=(const type& c) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] += c;
    }
    return *this;
}

template<typename type, uint32_t n, uint32_t m>
baseMatrix<type, n, m>& baseMatrix<type,n,m>::operator-=(const type& c) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] -= c;
    }
    return *this;
}

template<typename type, uint32_t n, uint32_t m>
baseMatrix<type, n, m>& baseMatrix<type,n,m>::operator*=(const type& c) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] *= c;
    }
    return *this;
}

template<typename type, uint32_t n, uint32_t m>
baseMatrix<type, n, m>& baseMatrix<type,n,m>::operator/=(const type& c) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] /= c;
    }
    return *this;
}

template<typename T, uint32_t N, uint32_t M> baseMatrix<T,N,M> operator+(const T& c, const baseMatrix<T,N,M>& other) {
    baseMatrix<T,N,M> result;
    for(uint32_t i = 0; i < N; i++) {
        result[i] = c + other[i];
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M> baseMatrix<T,N,M> operator-(const T& c, const baseMatrix<T,N,M>& other) {
    baseMatrix<T,N,M> result;
    for(uint32_t i = 0; i < N; i++) {
        result[i] = c - other[i];
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M> baseMatrix<T,N,M> operator*(const T& c, const baseMatrix<T,N,M>& other) {
    baseMatrix<T,N,M> result;
    for(uint32_t i = 0; i < N; i++) {
        result[i] = c * other[i];
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M> baseMatrix<T,N,M> operator+(const baseMatrix<T,N,M>& other, const T& c) {
    baseMatrix<T,N,M> result(other);
    for(uint32_t i = 0; i < N; i++) {
        result[i] += c;
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M> baseMatrix<T,N,M> operator-(const baseMatrix<T,N,M>& other, const T& c) {
    baseMatrix<T,N,M> result(other);
    for(uint32_t i = 0; i < N; i++) {
        result[i] -= c;
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M> baseMatrix<T,N,M> operator*(const baseMatrix<T,N,M>& other, const T& c) {
    baseMatrix<T,N,M> result(other);
    for(uint32_t i = 0; i < N; i++) {
        result[i] *= c;
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M, uint32_t K> baseMatrix<T,N,K> operator*(const baseMatrix<T,N,M>& left, const baseMatrix<T,M,K>& right) {
    baseMatrix<T,N,K> result;
    for(uint32_t i = 0; i < N; i++) {
        for(uint32_t j = 0; j < M; j++) {
            for(uint32_t k = 0; k < K; k++) {
                result[i][k] += left[i][j] * right[j][k];
            }
        }
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M> std::ostream& operator<<(std::ostream& out, const baseMatrix<T,N,M>& other) {
    for(uint32_t i = 0; i < N; i++) {
        std::cout << other.mat[i] << '\n';
    }
    return out;
}

template<typename type, uint32_t n, uint32_t m>
baseMatrix<type, n-1, m-1> baseMatrix<type,n,m>::extract(uint32_t i, uint32_t j) const {
    baseMatrix<type, n-1, m-1> result;
    for(uint32_t it = 0, ie = 0; it < n; it++){
        if(it != i){
            for(uint32_t jt = 0, je = 0; jt < m; jt++){
                if(jt != j){
                    result[ie][je] = mat[it][jt];
                    je++;
                }
            }
            ie++;
        }
    }
    return result;
}

template<typename T, uint32_t M, uint32_t N> baseMatrix<T,N,M> transpose(const baseMatrix<T,N,M>& other) {
    baseMatrix<T,M,N> result;
    for(uint32_t i = 0; i < N; i++) {
        for(uint32_t j = 0; j < M; j++) {
            result[j][i] = other[i][j];
        }
    }
    return result;
}

template<typename type, uint32_t n, uint32_t m> class matrix;

template<typename type>
class matrix<type, 2, 2> : public baseMatrix<type, 2, 2>
{
private:
    static constexpr uint32_t n = 2;

public:
    matrix(const vector<type, n>& v0, const vector<type, n>& v1) {
        this->mat[0] = v0;
        this->mat[1] = v1;
    }
    matrix(
        const type& m00, const type& m01,
        const type& m10, const type& m11
    ) {
        this->mat[0] = vector<type,n>{m00, m01};
        this->mat[1] = vector<type,n>{m10, m11};
    }
    matrix(const baseMatrix<type,n,n>& other) {
        for(uint32_t i = 0; i < n; i++){
            this->mat[i] = other[i];
        }
    }
    matrix<type,n,n>& operator=(const baseMatrix<type,n,n>& other) {
        for(uint32_t i = 0; i < n; i++){
            this->mat[i] = other[i];
        }
        return *this;
    }
};

template<typename type>
class matrix<type, 3, 3> : public baseMatrix<type, 3, 3>
{
private:
    static constexpr uint32_t n = 3;

public:
    matrix() : baseMatrix<type,n,n>() {}
    matrix(const vector<type, n>& v0, const vector<type, n>& v1, const vector<type, n>& v2) {
        this->mat[0] = v0;
        this->mat[1] = v1;
        this->mat[2] = v2;
    }
    matrix(const type& m) {
        this->mat[0] = vector<type,n>{m, 0.0f, 0.0f};
        this->mat[1] = vector<type,n>{0.0f, m, 0.0f};
        this->mat[2] = vector<type,n>{0.0f, 0.0f, m};
    }
    matrix(
        const type& m00, const type& m01, const type& m02,
        const type& m10, const type& m11, const type& m12,
        const type& m20, const type& m21, const type& m22
    ) {
        this->mat[0] = vector<type,n>{m00, m01, m02};
        this->mat[1] = vector<type,n>{m10, m11, m12};
        this->mat[2] = vector<type,n>{m20, m21, m22};
    }
    matrix(const baseMatrix<type,n,n>& other) {
        for(uint32_t i = 0; i < n; i++){
            this->mat[i] = other[i];
        }
    }
    matrix<type,n,n>& operator=(const baseMatrix<type,n,n>& other) {
        for(uint32_t i = 0; i < n; i++){
            this->mat[i] = other[i];
        }
        return *this;
    }
};

template<typename type>
class matrix<type, 4, 4> : public baseMatrix<type, 4, 4>
{
private:
    static constexpr uint32_t n = 4;

public:
    matrix() : baseMatrix<type,n,n>() {}
    matrix(std::ifstream& file) {
        for(uint32_t i = 0; i < n; i++){
            for(uint32_t j = 0; j < n; j++){
                file >> this->mat[i][j];
            }
        }
    }
    matrix(const type& m) {
        this->mat[0] = vector<type,n>{m, 0.0f, 0.0f, 0.0f};
        this->mat[1] = vector<type,n>{0.0f, m, 0.0f, 0.0f};
        this->mat[2] = vector<type,n>{0.0f, 0.0f, m, 0.0f};
        this->mat[3] = vector<type,n>{0.0f, 0.0f, 0.0f, m};
    }
    matrix(const vector<type, n>& v0, const vector<type, n>& v1, const vector<type, n>& v2, const vector<type, n>& v3) {
        this->mat[0] = v0;
        this->mat[1] = v1;
        this->mat[2] = v2;
        this->mat[3] = v3;
    }
    matrix(
        const type& m00, const type& m01, const type& m02, const type& m03,
        const type& m10, const type& m11, const type& m12, const type& m13,
        const type& m20, const type& m21, const type& m22, const type& m23,
        const type& m30, const type& m31, const type& m32, const type& m33
    ) {
        this->mat[0] = vector<type,n>{m00, m01, m02, m03};
        this->mat[1] = vector<type,n>{m10, m11, m12, m13};
        this->mat[2] = vector<type,n>{m20, m21, m22, m23};
        this->mat[3] = vector<type,n>{m30, m31, m32, m33};
    }
    matrix(const baseMatrix<type,n,n>& other) {
        for(uint32_t i = 0; i < n; i++){
            this->mat[i] = other[i];
        }
    }
    matrix<type,n,n>& operator=(const baseMatrix<type,n,n>& other) {
        for(uint32_t i = 0; i < n; i++){
            this->mat[i] = other[i];
        }
        return *this;
    }
};

template<typename type>
type det(const matrix<type,2,2>& m) {
    return  m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

template<typename type, uint32_t n>
type det(const matrix<type,n,n>& m) {
    type result = type(0);
    for(uint32_t i = 0; i < n; i++){
        result += (i % 2 ? -1.0f : 1.0f) * m[0][i] * det(matrix<type,n-1,n-1>(m.extract(0,i)));
    }
    return  result;
}

template<typename type, uint32_t n>
matrix<type,n,n> inverse(const matrix<type,n,n>& m) {
    matrix<type,n,n> result;
    for(uint32_t i = 0; i < n; i++){
        for(uint32_t j = 0; j < n; j++){
            result[i][j] = ((i + j) % 2 ? -1.0f : 1.0f) * det(matrix<type,n-1,n-1>(m.extract(i,j)));
        }
    }
    return transpose(result) / det(m);
}

template<typename type>
matrix<type,4,4> translate(vector<type,3> tr) {
    matrix<float,4,4> m{1.0f};
    m[0][3] += tr[0];
    m[1][3] += tr[1];
    m[2][3] += tr[2];
    return m;
}

template<typename type>
matrix<type,4,4> scale(vector<type,3> sc) {
    matrix<float,4,4> m{1.0f};
    m[0][0] *= sc[0];
    m[1][1] *= sc[1];
    m[2][2] *= sc[2];
    return m;
}

template<typename type>
matrix<type,4,4> perspective(const type& fovy, const type& aspect, const type& n, const type& f) {
    const type a = type(1) / std::tan(fovy / type(2));

    matrix<type,4,4> m(0.0f);
    m[0][0] = a / aspect;
    m[1][1] = - a;
    m[2][2] = (f + n) / (n - f);
    m[2][3] = type(2) * f * n / (n - f);
    m[3][2] = - type(1);

    return m;
}

template<typename type>
matrix<type,4,4> perspective(const type& fovy, const type& aspect, const type& n = type(0)) {
    const type a = type(1) / std::tan(fovy / type(2));

    matrix<type,4,4> m(0.0f);
    m[0][0] = a / aspect;
    m[1][1] = - a;
    m[2][2] = - type(1);
    m[2][3] = - type(2) * n;
    m[3][2] = - type(1);

    return m;
}

template<typename type>
type radians(const type& angle) {
    return type(M_PI) * angle / type(180);
}

#endif // MATRIX_H
