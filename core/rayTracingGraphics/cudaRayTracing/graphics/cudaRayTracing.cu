#include "cudaRayTracing.h"
#include "operations.h"
#include "ray.h"
#include "material.h"
#include <iostream>
namespace cuda {

cudaRayTracing::cudaRayTracing(){}
cudaRayTracing::~cudaRayTracing(){}

void cudaRayTracing::create()
{
    record = cuda::buffer<frameRecord>(width * height);
    baseColor = cuda::buffer<uint32_t>(width * height);
    bloomColor = cuda::buffer<uint32_t>(width * height);
}

void cudaRayTracing::buildTree(){
    devContainer = cuda::make_devicep<container_dev>(container_dev());

    hostTree = kdTree_host(hostContainer.begin(), hostContainer.size());
    maxDepth = findMaxDepth(&hostTree, maxDepth);
    std::vector<hitable*> hitables;
    for(const auto& p : hostContainer){
        hitables.push_back(p->hit());
    }
    add(devContainer.get(), hitables);

    std::vector<uint32_t> nodeCounter;
    buildSizesVector(&hostTree, nodeCounter);
    buffer<uint32_t> devNodeCounter(nodeCounter.size(), (uint32_t*) nodeCounter.data());

    cudaDeviceSetLimit(cudaLimitStackSize, 1024*5);
    if(std::is_same<container_dev, kdTree>::value){
        makeTree((kdTree*)devContainer.get(), devNodeCounter.get());
    }
}

__device__ bool isEmit(const cuda::hitRecord& rec){
    return (rec.rayDepth == 1 && rec.props.emissionFactor >= 0.98f) || (rec.scattering.getDirection().length2() > 0.0f && rec.lightIntensity >= 0.95f);
}

struct frameBuffer {
    vec4f base{0.0f};
    vec4f bloom{0.0f};
};

template<typename ContainerType>
__device__ frameBuffer getFrame(uint32_t minRayIterations, uint32_t maxRayIterations, cuda::camera* cam, float u, float v, cuda::hitRecord& rec, ContainerType* container, curandState* randState) {
    frameBuffer result;
    do {
        ray r = rec.rayDepth++ ? rec.scattering : cam->getPixelRay(u, v, randState);
        if (hitCoords coords; container->hit(r, coords)) {
            if(vec4 color = rec.color; coords.check()){
                coords.obj->calcHitRecord(r, coords, rec);
                rec.lightIntensity *= rec.props.absorptionFactor;
                rec.color = min(vec4f(rec.lightIntensity * rec.color.x(), rec.lightIntensity * rec.color.y(), rec.lightIntensity * rec.color.z(), rec.color.a()), color);
            }
        }

        vec4f scattering = scatter(r, rec.normal, rec.props, randState);
        if(scattering.length2() == 0.0f || rec.rayDepth >= maxRayIterations){
            result.base = rec.props.emissionFactor >= 0.98f ? rec.color : vec4f(0.0f, 0.0f, 0.0f, 1.0f);
            result.bloom = isEmit(rec) ? rec.color : vec4f(0.0f, 0.0f, 0.0f, 0.0f);
            rec = cuda::hitRecord{};
            break;
        }
        rec.scattering = ray(rec.point, scattering);
    } while (rec.rayDepth < minRayIterations);
    return result;
}

template <typename ContainerType>
__global__ void render(bool clear, size_t width, size_t height, size_t minRayIterations, size_t maxRayIterations, uint32_t* baseColor, uint32_t* bloomColor, frameRecord* record, cuda::camera* cam, ContainerType* container)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (int pixel = j * width + i; (i < width) && (j < height)) {
        curandState randState;
        curand_init(clock64(), pixel, 0, &randState);

        float u = 1.0f - 2.0f * float(i) / float(width);
        float v = 2.0f * float(j) / float(height) - 1.0f;

        if(clear){
            record[pixel] = frameRecord{};
        }

        frameBuffer frame = getFrame(minRayIterations, maxRayIterations, cam, u, v, record[pixel].hit, container, &randState);
        record[pixel].color += frame.base;
        record[pixel].bloom += frame.bloom;

        vec4f base = record[pixel].color / ::max(1.0f, record[pixel].color.a());
        baseColor[pixel] = uint32_t(255.0f*base[2]) << 0 | uint32_t(255.0f*base[1]) << 8 | uint32_t(255.0f*base[0]) << 16 | uint32_t(255) << 24;
        vec4f bloom = record[pixel].bloom / ::max(1.0f, record[pixel].bloom.a());
        bloomColor[pixel] = uint32_t(255.0f*bloom[2]) << 0 | uint32_t(255.0f*bloom[1]) << 8 | uint32_t(255.0f*bloom[0]) << 16 | uint32_t(255) << 24;
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

bool cudaRayTracing::calculateImage(uint32_t* hostBaseColor, uint32_t* hostBloomColor)
{
    dim3 blocks(width / xThreads + 1, height / yThreads + 1, 1);
    dim3 threads(xThreads, yThreads, 1);
    render<<<blocks, threads>>>(
        clear,
        width,
        height,
        minRayIterations,
        maxRayIterations,
        baseColor.get(),
        bloomColor.get(),
        record.get(),
        cam->get(),
        devContainer.get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    clear = false;

    checkCudaErrors(cudaMemcpy(hostBaseColor, baseColor.get(), sizeof(uint32_t) * baseColor.getSize(), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hostBloomColor, bloomColor.get(), sizeof(uint32_t) * bloomColor.getSize(), cudaMemcpyDeviceToHost));

    return true;
}

}
