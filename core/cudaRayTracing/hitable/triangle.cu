#include "triangle.h"
#include "operations.h"

namespace {
    __host__ __device__ float det3(const vec4& a, const vec4& b, const vec4& c) {
        return a.x() * b.y() * c.z() + b.x() * c.y() * a.z() + c.x() * a.y() * b.z() -
            (a.x() * c.y() * b.z() + b.x() * a.y() * c.z() + c.x() * b.y() * a.z());
    }
}

__host__ __device__ bool triangle::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    bool result = false;

    const vec4 a = vertexBuffer[index1].point - r.getOrigin();
    const vec4 b = vertexBuffer[index1].point - vertexBuffer[index2].point;
    const vec4 c = vertexBuffer[index1].point - vertexBuffer[index0].point;
    const vec4 d = r.getDirection();

    float det = det3(d, b, c);
    if (det != 0.0f) {
        const float t = det3(a, b, c) / det;
        const float u = det3(d, a, c) / det;
        const float v = det3(d, b, a) / det;
        const float s = 1.0f - u - v;

        result = (u >= 0.0f && v >= 0.0f && s >= 0.0f) && (t < tMax && t > tMin);
        if (result) {
            rec.t = t;
            rec.point = r.point(rec.t);
            rec.normal = normal(v * vertexBuffer[index0].normal + u * vertexBuffer[index2].normal + s * vertexBuffer[index1].normal);
            rec.color = v * vertexBuffer[index0].color + u * vertexBuffer[index2].color + s * vertexBuffer[index1].color;

            rec.props = {
                v * vertexBuffer[index0].props.refractiveIndex + u * vertexBuffer[index2].props.refractiveIndex + s * vertexBuffer[index1].props.refractiveIndex,
                v * vertexBuffer[index0].props.refractProb + u * vertexBuffer[index2].props.refractProb + s * vertexBuffer[index1].props.refractProb,
                v * vertexBuffer[index0].props.fuzz + u * vertexBuffer[index2].props.fuzz + s * vertexBuffer[index1].props.fuzz,
                v * vertexBuffer[index0].props.angle + u * vertexBuffer[index2].props.angle + s * vertexBuffer[index1].props.angle,
                v * vertexBuffer[index0].props.emissionFactor + u * vertexBuffer[index2].props.emissionFactor + s * vertexBuffer[index1].props.emissionFactor,
            };
        }
    }

    return result;
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

    triangle* hosttr = new triangle;
    checkCudaErrors(cudaMemcpy(&hosttr, tr, sizeof(triangle*), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(tr));

    return hosttr;
}
