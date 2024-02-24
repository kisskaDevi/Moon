#include "sphere.h"
#include "operations.h"

__host__ __device__ bool sphere::hit(const ray& r, float tMin, float tMax, hitCoords& coord) const {
    vec4 oc = r.getOrigin() - center;
    float a = 1.0f / r.getDirection().length2();
    float b = - dot(oc, r.getDirection()) * a;
    float c = oc.length2() - radius * radius * a;
    float discriminant = b * b - c;

    if (discriminant < 0) {
        return false;
    }

    discriminant = sqrt(discriminant);
    float temp = b - discriminant;
    bool result = (temp < tMax && temp > tMin);
    if (!result) {
        temp = b + discriminant;
        result = (temp < tMax && temp > tMin);
    }
    if (result) {
        coord = {temp, 0.0f, 0.0f};
    }
    return result;
}

__host__ __device__ hitRecord sphere::calcHitRecord(const ray& r, const hitCoords& coord) const {
    hitRecord rec;
    rec.point = r.point(coord.t);
    rec.normal = (rec.point - center) / radius;
    rec.color = color;
    rec.props = props;
    return rec;
}

__global__ void createSphere(sphere** sph, vec4 cen, float r, vec4 color, const properties props) {
    *sph = new sphere(cen, r, color, props);
}

sphere* sphere::create(vec4 cen, float r, vec4 color, const properties& props) {
    sphere** sph;
    checkCudaErrors(cudaMalloc((void**)&sph, sizeof(sphere**)));

    createSphere<<<1,1>>>(sph, cen, r, color, props);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    sphere* hostsph = nullptr;
    checkCudaErrors(cudaMemcpy(&hostsph, sph, sizeof(sphere*), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(sph));

    return hostsph;
}
