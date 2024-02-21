#include "triangle.h"
#include "operations.h"

namespace {
    __host__ __device__ float det3(const vec4& a, const vec4& b, const vec4& c) {
        return a.x() * b.y() * c.z() + b.x() * c.y() * a.z() + c.x() * a.y() * b.z() -
            (a.x() * c.y() * b.z() + b.x() * a.y() * c.z() + c.x() * b.y() * a.z());
    }
}

__host__ __device__ bool triangle::hit(const ray& r, float tMin, float tMax, hitCoords& rec) const {
    const vec4 a = vertexBuffer[index1].point - r.getOrigin();
    const vec4 b = vertexBuffer[index1].point - vertexBuffer[index2].point;
    const vec4 c = vertexBuffer[index1].point - vertexBuffer[index0].point;
    const vec4 d = r.getDirection();

    float det = det3(d, b, c);
    if (det == 0.0f) {
        return false;
    }

    const float t = det3(a, b, c) / det;
    const float u = det3(d, a, c) / det;
    const float v = det3(d, b, a) / det;
    const float s = 1.0f - u - v;

    bool result = (u >= 0.0f && v >= 0.0f && s >= 0.0f) && (t < tMax && t > tMin);
    if (result) {
        rec = {t,u,v,s};
    }

    return result;
}

__host__ __device__ void triangle::calcHitRecord(const ray& r, const hitCoords& coord, hitRecord& rec) const {
    rec.point = r.point(coord.t);
    rec.normal = normal(coord.v * vertexBuffer[index0].normal + coord.u * vertexBuffer[index2].normal + coord.s * vertexBuffer[index1].normal);
    rec.color = coord.v * vertexBuffer[index0].color + coord.u * vertexBuffer[index2].color + coord.s * vertexBuffer[index1].color;

    rec.props = {
        coord.v * vertexBuffer[index0].props.refractiveIndex + coord.u * vertexBuffer[index2].props.refractiveIndex + coord.s * vertexBuffer[index1].props.refractiveIndex,
        coord.v * vertexBuffer[index0].props.refractProb + coord.u * vertexBuffer[index2].props.refractProb + coord.s * vertexBuffer[index1].props.refractProb,
        coord.v * vertexBuffer[index0].props.fuzz + coord.u * vertexBuffer[index2].props.fuzz + coord.s * vertexBuffer[index1].props.fuzz,
        coord.v * vertexBuffer[index0].props.angle + coord.u * vertexBuffer[index2].props.angle + coord.s * vertexBuffer[index1].props.angle,
        coord.v * vertexBuffer[index0].props.emissionFactor + coord.u * vertexBuffer[index2].props.emissionFactor + coord.s * vertexBuffer[index1].props.emissionFactor,
    };
}

__global__ void createTriangle(triangle** tr, const size_t i0, const size_t i1, const size_t i2, vertex* vertexBuffer) {
    *tr = new triangle(i0, i1, i2, vertexBuffer);
}

triangle* triangle::create(const size_t& i0, const size_t& i1, const size_t& i2, vertex* vertexBuffer) {
    triangle** tr;
    checkCudaErrors(cudaMalloc((void**)&tr, sizeof(triangle**)));

    createTriangle<<<1,1>>>(tr, i0, i1, i2, vertexBuffer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    triangle* hosttr = nullptr;
    checkCudaErrors(cudaMemcpy(&hosttr, tr, sizeof(triangle*), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(tr));

    return hosttr;
}