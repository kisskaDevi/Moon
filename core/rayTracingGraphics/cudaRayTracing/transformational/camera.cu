#include "camera.h"

__host__ __device__ void cuda::camera::update(){
    horizontal = aspect * vec4::getHorizontal(viewRay.getDirection());
    vertical = vec4::getVertical(viewRay.getDirection());
}

__host__ __device__ cuda::camera::camera(){}

__host__ __device__ cuda::camera::camera(
    const ray viewRay,
    float aspect,
    float matrixScale,
    float matrixOffset,
    float focus,
    float apertura) :
    viewRay(viewRay),
    aspect(aspect),
    matrixScale(matrixScale),
    matrixOffset(matrixOffset),
    focus(focus),
    apertura(apertura)
{
    update();
}

__host__ __device__ cuda::camera::camera(const ray& viewRay, float aspect) : viewRay(viewRay), aspect(aspect){
    update();
}

__device__ ray cuda::camera::getPixelRay(float u, float v, curandState* local_rand_state) const {
    const float t = focus / (matrixOffset - focus);
    u = matrixScale * t * u + apertura * float(curand_uniform(local_rand_state));
    v = matrixScale * t * v + apertura * float(curand_uniform(local_rand_state));
    return ray(viewRay.point(matrixOffset), t * matrixOffset * viewRay.getDirection() - (u * horizontal + v * vertical));
}

__host__ __device__ ray cuda::camera::getPixelRay(float u, float v) const {
    const float t = focus / (matrixOffset - focus);
    u = matrixScale * t * u;
    v = matrixScale * t * v;
    return ray(viewRay.point(matrixOffset), t * matrixOffset * viewRay.getDirection() - (u * horizontal + v * vertical));
}
