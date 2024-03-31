#include "cudaRayTracing.h"
#include "operations.h"
#include "ray.h"
#include "material.h"

namespace cuda {

cudaRayTracing::cudaRayTracing() :
    devContainer(cuda::make_devicep<container_dev>(container_dev()))
{}

cudaRayTracing::~cudaRayTracing(){
    if(hostTree){
        delete hostTree;
    }
}

void cudaRayTracing::create()
{
    frame = cuda::buffer<frameBuffer>(width * height);
    swapChainImage = cuda::buffer<uint32_t>(width * height);
}

__global__ void createTree(cudaRayTracing::kdTree_dev* devTree, cudaRayTracing::container_dev* container)
{
    new (devTree) cudaRayTracing::kdTree_dev(container->begin(), container->size());
}

void cudaRayTracing::buildTree(){
    hostTree = new kdTree_host(hostContainer.begin(), hostContainer.size());
    std::vector<hitable*> hitables;
    for(const auto& p : hostContainer){
        hitables.push_back(p->hit());
    }
    add(devContainer.get(), hitables);

    devTree = make_devicep<kdTree_dev>(kdTree_dev(nullptr, 0));
    createTree<<<1,1>>>(devTree.get(), devContainer.get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

template<bool bloom = false>
__device__ bool isEmit(const cuda::hitRecord& rec){
    return (rec.props.emissionFactor >= 0.98f) &&
           (!bloom || (rec.color.x() >= 0.9f || rec.color.y() >= 0.9f || rec.color.z() >= 0.9f));
}

template<typename ContainerType, bool bloom = false>
__device__ vec4f color(uint32_t minRayIterations, uint32_t maxRayIterations, cuda::camera* cam, float u, float v, cuda::hitRecord& rec, ContainerType* container, curandState* randState) {
    vec4f result = vec4f(0.0f);
    do {
        ray r = rec.rayDepth++ ? rec.scattering : cam->getPixelRay(u, v, randState);
        if constexpr (bloom) {
            r = ray(r.getOrigin(), random_in_unit_sphere(r.getDirection(), 0.05f * pi, randState));
        }
        if (hitCoords coords; container->hit(r, coords)) {
            if(vec4 color = rec.color; coords.check()){
                coords.obj->calcHitRecord(r, coords, rec);
                rec.lightIntensity *= rec.props.absorptionFactor;
                rec.color = min(vec4f(rec.lightIntensity * rec.color.x(), rec.lightIntensity * rec.color.y(), rec.lightIntensity * rec.color.z(), rec.color.a()), color);
            }
        }

        vec4f scattering = scatter(r, rec.normal, rec.props, randState);
        if(scattering.length2() == 0.0f || rec.rayDepth >= maxRayIterations){
            result = isEmit<bloom>(rec) ? rec.color : vec4f(0.0f, 0.0f, 0.0f, 1.0f);
            rec = cuda::hitRecord{};
            break;
        }
        rec.scattering = ray(rec.point, scattering);
    } while (rec.rayDepth < minRayIterations);
    return result;
}

template <typename ContainerType>
__global__ void render(bool clear, size_t width, size_t height, size_t minRayIterations, size_t maxRayIterations, uint32_t* dst, frameBuffer* frame, cuda::camera* cam, ContainerType* container)
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

        frame[pixel].base.color += color<ContainerType>(minRayIterations, maxRayIterations, cam, u, v, frame[pixel].base.record, container, &randState);
        frame[pixel].bloom.color += color<ContainerType, true>(minRayIterations, 2, cam, u, v, frame[pixel].bloom.record, container, &randState);

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
    render<<<blocks, threads>>>(
        clear,
        width,
        height,
        minRayIterations,
        maxRayIterations,
        swapChainImage.get(),
        frame.get(),
        cam->get(),
        devContainer.get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    clear = false;

    checkCudaErrors(cudaMemcpy(hostFrameBuffer, swapChainImage.get(), sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost));

    return true;
}

}
