#include "cudaRayTracing.h"
#include "operations.h"
#include "ray.h"
#include "material.h"
#include "hitableArray.h"

namespace cuda {

cudaRayTracing::cudaRayTracing() :
    container(cuda::make_devicep<cuda::hitableContainer>(cuda::hitableArray()))
{}

cudaRayTracing::~cudaRayTracing(){
}

void cudaRayTracing::create()
{
    frame = cuda::buffer<frameBuffer>(width * height);
    swapChainImage = cuda::buffer<uint32_t>(width * height);
}

template<bool bloom = false>
__device__ bool isEmit(const cuda::hitRecord& rec){
    return (rec.props.emissionFactor >= 0.98f) &&
           (!bloom || (rec.color.x() >= 0.9f || rec.color.y() >= 0.9f || rec.color.z() >= 0.9f));
}

template<bool bloom = false>
__device__ vec4f color(uint32_t minRayIterations, uint32_t maxRayIterations, cuda::camera* cam, float u, float v, cuda::hitRecord& rec, cuda::hitableContainer* container, curandState* randState) {
    vec4f result = vec4f(0.0f);
    do {
        ray r = rec.rayDepth++ ? rec.r : cam->getPixelRay(u, v, randState);
        if constexpr (bloom) {
            r = ray(r.getOrigin(), random_in_unit_sphere(r.getDirection(), 0.05f * pi, randState));
        }
        if (vec4f color = rec.color; container->hit(r, 0.001f, 1e+37, rec)) {
            rec.lightIntensity *= rec.props.absorptionFactor;
            rec.color = min(vec4f(rec.lightIntensity * rec.color.x(), rec.lightIntensity * rec.color.y(), rec.lightIntensity * rec.color.z(), rec.color.a()), color);
        }

        vec4f scattering = scatter(r, rec.normal, rec.props, randState);
        if(scattering.length2() == 0.0f || rec.rayDepth >= maxRayIterations){
            result= isEmit<bloom>(rec) ? rec.color : vec4f(0.0f, 0.0f, 0.0f, 1.0f);
            rec = cuda::hitRecord{};
            break;
        }
        rec.r = ray(rec.point, scattering);
    } while (rec.rayDepth < minRayIterations);
    return result;
}

__global__ void render(bool clear, size_t width, size_t height, size_t minRayIterations, size_t maxRayIterations, uint32_t* dst, frameBuffer* frame, cuda::camera* cam, cuda::hitableContainer* container)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (int pixel = j * width + i; (i < width) && (j < height)) {
        curandState randState;
        curand_init(clock64(), pixel, 0, &randState);

        float u = 1.0f - 2.0f * float(i) / float(width);
        float v = 2.0f * float(j) / float(height) - 1.0f;

        if(clear){
            frame[pixel] = frameBuffer{};
        }

        frame[pixel].base.color += color(minRayIterations, maxRayIterations, cam, u, v, frame[pixel].base.record, container, &randState);
        frame[pixel].bloom.color += color<true>(minRayIterations, 2, cam, u, v, frame[pixel].bloom.record, container, &randState);

        vec4f base = frame[pixel].base.color / ::max(1.0f, frame[pixel].base.color.a());
        vec4f bloom = frame[pixel].bloom.color / ::max(1.0f, frame[pixel].bloom.color.a());
        vec4f res = max(base, bloom);
        dst[pixel] = (uint32_t(255.0f*res[2]) << 0 | uint32_t(255.0f*res[1]) << 8 | uint32_t(255.0f*res[0]) << 16 | uint32_t(255) << 24);
    }
}

__global__ void updateKernel(cuda::camera* cam){
    cam->update();
}

void cudaRayTracing::update(){
    updateKernel<<<1, 1>>>(cam->get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

bool cudaRayTracing::calculateImage(uint32_t* hostFrameBuffer)
{
    dim3 blocks(width / xThreads + 1, height / yThreads + 1, 1);
    dim3 threads(xThreads, yThreads, 1);
    render<<<blocks, threads>>>(clear, width, height, minRayIterations, maxRayIterations, swapChainImage.get(), frame.get(), cam->get(), container());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    clear = false;

    checkCudaErrors(cudaMemcpy(hostFrameBuffer, swapChainImage.get(), sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost));

    return true;
}

}
