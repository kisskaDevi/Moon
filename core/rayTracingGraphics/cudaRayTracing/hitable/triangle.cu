#include "triangle.h"
#include "operations.h"

namespace cuda::rayTracing {

__host__ __device__ Triangle::Triangle(const size_t& i0, const size_t& i1, const size_t& i2, const Vertex* vertexBuffer, cudaTextureObject_t texture)
    : index{i0,i1,i2}, vertexBuffer(vertexBuffer), texture(texture)
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

#define interpolate(coord, val) coord.v * vertexBuffer[index[0]].val + coord.u * vertexBuffer[index[2]].val + s * vertexBuffer[index[1]].val

__device__ void Triangle::calcHitRecord(const ray& r, const HitCoords& coord, HitRecord& rec) const {
    const float s = 1.0f - coord.u - coord.v;
    rec.vertex.point = r.point(coord.tmax);
    rec.vertex.normal = normal(interpolate(coord, normal));
    rec.vertex.u = interpolate(coord, u);
    rec.vertex.v = interpolate(coord, v);
    uchar4 texv = {0, 0, 0, 0};
    if(texture != 0){
        texv = tex2D<uchar4>(texture, rec.vertex.u, 1.0f - rec.vertex.v);
    }
    rec.vertex.color = texture == 0 ? interpolate(coord, color) :
        vec4f(
            (float)texv.x / 255.0f,
            (float)texv.y / 255.0f,
            (float)texv.z / 255.0f,
            1.0f);
    rec.vertex.props = {
        interpolate(coord, props.refractiveIndex),
        interpolate(coord, props.refractProb),
        interpolate(coord, props.fuzz),
        interpolate(coord, props.angle),
        interpolate(coord, props.emissionFactor),
        interpolate(coord, props.absorptionFactor)
    };
}

__global__ void createKernel(Triangle* tr, const size_t i0, const size_t i1, const size_t i2, const Vertex* vertexBuffer, cudaTextureObject_t texture) {
    tr = new (tr) Triangle(i0, i1, i2, vertexBuffer, texture);
}

void Triangle::create(Triangle* dpointer, const Triangle& host){
    createKernel<<<1,1>>>(dpointer, host.index[0], host.index[1], host.index[2], host.vertexBuffer, host.texture);
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
