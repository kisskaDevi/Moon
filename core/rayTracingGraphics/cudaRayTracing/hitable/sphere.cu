#include "sphere.h"
#include "operations.h"

namespace cuda {

__host__ __device__ sphere::sphere(const vec4f& cen, float r, const vec4f& color, const properties& props) : center(cen), radius(r), color(color), props(props) {}

__host__ __device__ sphere::sphere(const vec4f& cen, float r, const vec4f& color) : center(cen), radius(r), color(color) {}

__host__ __device__ bool sphere::hit(const ray& r, hitCoords& coord) const {
    vec4f oc = r.getOrigin() - center;
    float a = 1.0f / r.getDirection().length2();
    float b = - dot(oc, r.getDirection()) * a;
    float c = oc.length2() - radius * radius * a;
    float discriminant = b * b - c;

    if (discriminant < 0) {
        return false;
    }

    discriminant = sqrt(discriminant);
    float temp = b - discriminant;
    bool result = (temp < coord.tmax && temp > coord.tmin);
    if (!result) {
        temp = b + discriminant;
        result = (temp < coord.tmax && temp > coord.tmin);
    }
    if (result) {
        coord.tmax = temp;
    }
    return result;
}

__host__ __device__ void sphere::calcHitRecord(const ray& r, const hitCoords& coord, hitRecord& rec) const {
    rec.point = r.point(coord.tmax);
    rec.normal = (rec.point - center) / radius;
    rec.color = color;
    rec.props = props;
}

__global__ void createKernel(sphere* sph, vec4f cen, float r, vec4f color, const properties props) {
    sph = new (sph) sphere(cen, r, color, props);
}

void sphere::create(sphere* dpointer, const sphere& host){
    createKernel<<<1,1>>>(dpointer, host.center, host.radius, host.color, host.props);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(sphere* p) {
    p->~sphere();
}

void sphere::destroy(sphere* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ __device__ box sphere::calcBox() const {
    box bbox;
    bbox.min = center - vec4f(radius, radius, radius, 0.0f);
    bbox.max = center + vec4f(radius, radius, radius, 0.0f);
    return bbox;
}
}
