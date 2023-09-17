#include "sphere.h"

__device__ bool sphere::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    bool result = false;

    vec4 oc = r.getOrigin() - center;
    float a = dot(r.getDirection(), r.getDirection());
    float b = dot(oc, r.getDirection()) / a;
    float c = dot(oc, oc) - radius * radius / a;
    float discriminant = b * b - c;

    if (discriminant >= 0) {
        discriminant = sqrt(discriminant);
        float temp = -b - discriminant;
        result = (temp < tMax && temp > tMin);
        if (!result) {
            temp = -b + discriminant;
            result = (temp < tMax && temp > tMin);
        }
        if (result) {
            rec.t = temp;
            rec.point = r.point(rec.t);
            rec.normal = (rec.point - center) / radius;
            rec.color = color;
            rec.props = props;
        }
    }
    return result;
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

    sphere* hostsph = new sphere;
    checkCudaErrors(cudaMemcpy(&hostsph, sph, sizeof(sphere*), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(sph));

    return hostsph;
}
