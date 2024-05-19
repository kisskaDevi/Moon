#include "cudaRayTracing.h"
#include "operations.h"
#include "ray.h"
#include "material.h"

namespace cuda::rayTracing {

RayTracing::RayTracing(){}
RayTracing::~RayTracing(){}


__global__ void initCurandState(size_t width, size_t height, curandState* randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (int pixel = j * width + i; (i < width) && (j < height)) {
        curand_init(clock64(), pixel, 0, &randState[pixel]);
    }
}

void RayTracing::create()
{
    record = Buffer<FrameRecord>(width * height);
    baseColor = Buffer<uint32_t>(width * height);
    bloomColor = Buffer<uint32_t>(width * height);
    randState = Buffer<curandState>(width * height);

    dim3 blocks(width / xThreads + 1, height / yThreads + 1, 1);
    dim3 threads(xThreads, yThreads, 1);
    initCurandState<<<blocks, threads>>>(width, height, randState.get());
}

void RayTracing::buildTree(){
    hostContainer.makeTree();

    devContainer = make_devicep<Container_dev>(Container_dev());
    add(devContainer.get(), extractHitables(hostContainer.storage));

    if(std::is_same<Container_dev, HitableKDTree>::value){
        const auto linearSizes = hostContainer.getLinearSizes();
        const auto linearBoxes = hostContainer.getLinearBoxes();
        Buffer<uint32_t> devNodeCounter(linearSizes.size(), (uint32_t*) linearSizes.data());
        Buffer<box> devNodeBox(linearBoxes.size(), (box*) linearBoxes.data());
        makeTree((HitableKDTree*)devContainer.get(), devNodeCounter.get(), devNodeBox.get(), linearSizes.size());
    }
}

__device__ bool isEmit(const HitRecord& rec){
    return (rec.rayDepth == 1 && rec.vertex.props.emissionFactor >= 0.98f) || (rec.scattering.getDirection().length2() > 0.0f && rec.lightIntensity >= 0.95f);
}

struct FrameBuffer {
    vec4f base{0.0f};
    vec4f bloom{0.0f};
};

template<typename ContainerType>
__device__ FrameBuffer getFrame(uint32_t minRayIterations, uint32_t maxRayIterations, Camera* cam, float u, float v, HitRecord& rec, ContainerType* container, curandState* randState) {
    FrameBuffer result;
    do {
        ray r = rec.rayDepth++ ? rec.scattering : cam->getPixelRay(u, v, randState);
        if (HitCoords coords; container->hit(r, coords)) {
            if(vec4 color = rec.vertex.color; coords.check()){
                coords.obj->calcHitRecord(r, coords, rec);
                rec.lightIntensity *= rec.vertex.props.absorptionFactor;
                rec.vertex.color = min(
                    vec4f(rec.lightIntensity * rec.vertex.color.x(),
                          rec.lightIntensity * rec.vertex.color.y(),
                          rec.lightIntensity * rec.vertex.color.z(),
                          rec.vertex.color.a()),
                    color);
            }
        }

        vec4f scattering = scatter(r, rec.vertex.normal, rec.vertex.props, randState);
        if(scattering.length2() == 0.0f || rec.rayDepth >= maxRayIterations){
            result.base = rec.vertex.props.emissionFactor >= 0.98f ? rec.vertex.props.emissionFactor * rec.vertex.color : vec4f(0.0f, 0.0f, 0.0f, 1.0f);
            result.bloom = isEmit(rec) ? rec.vertex.color : vec4f(0.0f, 0.0f, 0.0f, 0.0f);
            rec = HitRecord{};
            break;
        }
        rec.scattering = ray(rec.vertex.point, scattering);
    } while (rec.rayDepth < minRayIterations);
    return result;
}

template <typename ContainerType>
__global__ void render(bool clear, size_t width, size_t height, size_t minRayIterations, size_t maxRayIterations, uint32_t* baseColor, uint32_t* bloomColor, FrameRecord* record, Camera* cam, ContainerType* container, curandState* randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (int pixel = j * width + i; (i < width) && (j < height)) {
        // curandState randState;
        // curand_init(clock64(), pixel, 0, &randState);

        float u = 1.0f - 2.0f * float(i) / float(width);
        float v = 2.0f * float(j) / float(height) - 1.0f;

        if(clear){
            record[pixel] = FrameRecord{};
        }

        FrameBuffer frame = getFrame(minRayIterations, maxRayIterations, cam, u, v, record[pixel].hit, container, &randState[pixel]);
        record[pixel].color += frame.base;
        record[pixel].bloom += frame.bloom;

        vec4f base = record[pixel].color / ::max(1.0f, record[pixel].color.a());
        baseColor[pixel] = uint32_t(255.0f*base[2]) << 0 | uint32_t(255.0f*base[1]) << 8 | uint32_t(255.0f*base[0]) << 16 | uint32_t(255) << 24;
        vec4f bloom = record[pixel].bloom / ::max(1.0f, record[pixel].bloom.a());
        bloomColor[pixel] = uint32_t(255.0f*bloom[2]) << 0 | uint32_t(255.0f*bloom[1]) << 8 | uint32_t(255.0f*bloom[0]) << 16 | uint32_t(255) << 24;
    }
}

__global__ void updateKernel(Camera* cam){
    cam->update();
}

void RayTracing::update(){
    updateKernel<<<1, 1>>>(cam->get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

bool RayTracing::calculateImage(uint32_t* hostBaseColor, uint32_t* hostBloomColor)
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
        devContainer.get(),
        randState.get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    clear = false;

    checkCudaErrors(cudaMemcpy(hostBaseColor, baseColor.get(), sizeof(uint32_t) * baseColor.getSize(), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hostBloomColor, bloomColor.get(), sizeof(uint32_t) * bloomColor.getSize(), cudaMemcpyDeviceToHost));

    return true;
}

}
