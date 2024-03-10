#include "camera.h"
#include "operations.h"

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
    horizontal = aspect * vec4::getHorizontal(viewRay.getDirection());
    vertical = vec4::getVertical(viewRay.getDirection());
}

__host__ __device__ cuda::camera::camera(const ray viewRay, float aspect) : aspect(aspect){
    setViewRay(viewRay);
}

__device__ ray cuda::camera::getPixelRay(float u, float v, curandState* local_rand_state) {
    const float t = focus / (matrixOffset - focus);
    u = matrixScale * t * u + apertura * float(curand_uniform(local_rand_state));
    v = matrixScale * t * v + apertura * float(curand_uniform(local_rand_state));
    return ray(viewRay.point(matrixOffset), t * matrixOffset * viewRay.getDirection() - (u * horizontal + v * vertical));
}

__host__ __device__ ray cuda::camera::getPixelRay(float u, float v) {
    const float t = focus / (matrixOffset - focus);
    u = matrixScale * t * u;
    v = matrixScale * t * v;
    return ray(viewRay.point(matrixOffset), t * matrixOffset * viewRay.getDirection() - (u * horizontal + v * vertical));
}

__host__ __device__ ray cuda::camera::getViewRay(){
    return viewRay;
}

__host__ __device__ void cuda::camera::setViewRay(const ray& viewRay){
    this->viewRay = viewRay;
    horizontal = aspect * vec4::getHorizontal(viewRay.getDirection());
    vertical = vec4::getVertical(viewRay.getDirection());
}

__host__ __device__ void cuda::camera::setFocus(const float& focus){
    this->focus = focus;
}

cuda::camera* cuda::camera::create(const ray& viewRay, float aspect) {
    cuda::camera* cam = nullptr;
    cuda::camera hostcam(viewRay, aspect);
    checkCudaErrors(cudaMalloc((void**)&cam, sizeof(cuda::camera)));
    checkCudaErrors(cudaMemcpy(cam, &hostcam, sizeof(cuda::camera), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    return cam;
}

void cuda::camera::reset(camera* cam, const ray& viewRay, float aspect) {
    cuda::camera hostcam(viewRay, aspect);
    checkCudaErrors(cudaMemcpy(cam, &hostcam, sizeof(cuda::camera), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void cuda::camera::destroy(camera* cam) {
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void setViewRayKernel(cuda::camera* cam, const ray viewRay){
    cam->setViewRay(viewRay);
}

void cuda::camera::setViewRay(camera* cam, const ray& viewRay){
    setViewRayKernel<<<1,1>>>(cam, viewRay);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void setFocusKernel(cuda::camera* cam, const float focus){
    cam->setFocus(focus);
}

void cuda::camera::setFocus(camera* cam, const float& focus){
    setFocusKernel<<<1,1>>>(cam, focus);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

cuda::camera cuda::camera::copyToHost(camera* pDevice){
    cuda::camera hostcam;
    checkCudaErrors(cudaMemcpy(&hostcam, pDevice, sizeof(cuda::camera), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    return hostcam;
}
