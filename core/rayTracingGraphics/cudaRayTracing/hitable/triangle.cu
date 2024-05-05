#include "triangle.h"
#include "operations.h"

namespace cuda::rayTracing {

__host__ __device__ Triangle::Triangle(const size_t& i0, const size_t& i1, const size_t& i2, const Vertex* vertexBuffer)
    : index{i0,i1,i2}, vertexBuffer(vertexBuffer)
{};

__host__ __device__ bool Triangle::hit(const ray& r, HitCoords& coord) const {
    const vec4f a = vertexBuffer[index[1]].point - r.getOrigin();
    const vec4f b = vertexBuffer[index[1]].point - vertexBuffer[index[2]].point;
    const vec4f c = vertexBuffer[index[1]].point - vertexBuffer[index[0]].point;
    const vec4f d = r.getDirection();

    float det = det3(d, b, c);
    if (det == 0.0f) {
        return false;
    }
    det = 1.0f / det;

    const float t = det3(a, b, c) * det;
    if(t > coord.tmax || t < coord.tmin){
        return false;
    }

    const float u = det3(d, a, c) * det;
    const float v = det3(d, b, a) * det;
    const float s = 1.0f - u - v;

    if (u >= 0.0f && v >= 0.0f && s >= 0.0f) {
        coord = {coord.tmin, t, u, v};
        return true;
    }

    return false;
}

__host__ __device__ void Triangle::calcHitRecord(const ray& r, const HitCoords& coord, HitRecord& rec) const {
    const float s = 1.0f - coord.u - coord.v;
    rec.point = r.point(coord.tmax);
    rec.normal = normal(coord.v * vertexBuffer[index[0]].normal + coord.u * vertexBuffer[index[2]].normal + s * vertexBuffer[index[1]].normal);
    rec.color = coord.v * vertexBuffer[index[0]].color + coord.u * vertexBuffer[index[2]].color + s * vertexBuffer[index[1]].color;
    rec.props = {
        coord.v * vertexBuffer[index[0]].props.refractiveIndex + coord.u * vertexBuffer[index[2]].props.refractiveIndex + s * vertexBuffer[index[1]].props.refractiveIndex,
        coord.v * vertexBuffer[index[0]].props.refractProb + coord.u * vertexBuffer[index[2]].props.refractProb + s * vertexBuffer[index[1]].props.refractProb,
        coord.v * vertexBuffer[index[0]].props.fuzz + coord.u * vertexBuffer[index[2]].props.fuzz + s * vertexBuffer[index[1]].props.fuzz,
        coord.v * vertexBuffer[index[0]].props.angle + coord.u * vertexBuffer[index[2]].props.angle + s * vertexBuffer[index[1]].props.angle,
        coord.v * vertexBuffer[index[0]].props.emissionFactor + coord.u * vertexBuffer[index[2]].props.emissionFactor + s * vertexBuffer[index[1]].props.emissionFactor,
        coord.v * vertexBuffer[index[0]].props.absorptionFactor + coord.u * vertexBuffer[index[2]].props.absorptionFactor + s * vertexBuffer[index[1]].props.absorptionFactor
    };
}

__global__ void createKernel(Triangle* tr, const size_t i0, const size_t i1, const size_t i2, const Vertex* vertexBuffer) {
    tr = new (tr) Triangle(i0, i1, i2, vertexBuffer);
}

void Triangle::create(Triangle* dpointer, const Triangle& host){
    createKernel<<<1,1>>>(dpointer, host.index[0], host.index[1], host.index[2], host.vertexBuffer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(Triangle* p) {
    p->~Triangle();
}

void Triangle::destroy(Triangle* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ __device__ box Triangle::getBox() const {
    box bbox;
    bbox.min = min(vertexBuffer[index[0]].point, min(vertexBuffer[index[1]].point, vertexBuffer[index[2]].point));
    bbox.max = max(vertexBuffer[index[0]].point, max(vertexBuffer[index[1]].point, vertexBuffer[index[2]].point));
    return bbox;
}

}
