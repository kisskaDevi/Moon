#include "cudaRayTracing.h"
#include "operations.h"
#include "ray.h"
#include "material.h"

void cudaRayTracing::create()
{
    colorImage = cuda::buffer<vec4>(width * height);
    bloomImage = cuda::buffer<vec4>(width * height);
    swapChainImage  = cuda::buffer<uint32_t>(width * height);

    checkCudaErrors(cudaMalloc((void**)&randState, width * height * sizeof(curandState)));
}

void cudaRayTracing::destroy() {
    checkCudaErrors(cudaFree(randState));
    colorImage.destroy();
    bloomImage.destroy();
    swapChainImage.destroy();
}

namespace base {
__device__ vec4 color(ray r, size_t maxIterations, hitableContainer* container, curandState* local_rand_state) {
    vec4 color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
    hitRecord rec;

    for (; maxIterations > 0; maxIterations--) {
        if (r.getDirection().length2() > 0.0f && container->hit(r, 0.001f, 1e+37, rec)) {
            color = min(rec.color, color);
            r = ray(rec.point, scatter(r, rec.normal, rec.props, local_rand_state));
        } else {
            break;
        }
    }
    return rec.props.emissionFactor >= 1.0f ? vec4(color.x(), color.y(), color.z(), 1.0f) : vec4(0.0f, 0.0f, 0.0f, 0.0f);
}
}

namespace bloom {
__device__ bool isBloomed(const vec4& color, const hitRecord& rec){
    return (rec.props.emissionFactor >= 1.0f) && (color.x() >= 0.9f || color.y() >= 0.9f || color.z() >= 0.9f);
}

__device__ vec4 color(ray r, size_t maxIterations, hitableContainer* container, curandState* local_rand_state) {
    vec4 color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
    hitRecord rec;
    r = ray(r.getOrigin(), random_in_unit_sphere(r.getDirection(), 0.025 * pi, local_rand_state));

    for (; maxIterations > 0; maxIterations--) {
        if (r.getDirection().length2() > 0.0f && container->hit(r, 0.001f, 1e+37, rec)) {
            color = min(rec.color, color);
            r = ray(rec.point, scatter(r, rec.normal, rec.props, local_rand_state));
        } else {
            break;
        }
    }
    return isBloomed(color, rec) ? vec4(color.x(), color.y(), color.z(), 1.0f) : vec4(0.0f, 0.0f, 0.0f, 1.0f);
}
}

__global__ void render(bool clear, size_t width, size_t height, size_t rayDepth, curandState* randState, uint32_t* dst, vec4* base, vec4* bloom, cuda::camera* cam, hitableContainer* container)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (int pixel = j * width + i; (i < width) && (j < height)) {
        curand_init(clock64(), pixel, 0, &randState[pixel]);

        float u = 1.0f - 2.0f * float(i) / float(width);
        float v = 2.0f * float(j) / float(height) - 1.0f;

        if(clear){
            base[pixel] = vec4(0.0f, 0.0f, 0.0f, 0.0f);
            bloom[pixel] = vec4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        ray camRay = cam->getPixelRay(u, v, &randState[pixel]);
        base[pixel] += base::color(camRay, rayDepth, container, &randState[pixel]);
        bloom[pixel] += bloom::color(camRay, 2, container, &randState[pixel]);

        vec4 res = base[pixel] / (base[pixel].a() == 0.0f ? 1.0f : base[pixel].a());
        res += bloom[pixel] / (bloom[pixel].a() == 0.0f ? 1.0f : bloom[pixel].a());

        auto max = [](const float& a, const float& b) {
            return a >= b ? a : b;
        };

        float maximum =  max(res.r(), max(res.g(), res.b()));
        res /= (maximum > 1.0f) ? maximum : 1.0f;
        dst[pixel] = (uint32_t(255.0f*res[2]) << 0 | uint32_t(255.0f*res[1]) << 8 | uint32_t(255.0f*res[0]) << 16 | uint32_t(255) << 24);
    }
}

void cudaRayTracing::calculateImage(uint32_t* hostFrameBuffer)
{
    dim3 blocks(width / xThreads + 1, height / yThreads + 1, 1);
    dim3 threads(xThreads, yThreads, 1);

    render<<<blocks, threads>>>(clear, width, height, rayDepth, randState, swapChainImage.get(), colorImage.get(), bloomImage.get(), cam, container);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpy(hostFrameBuffer, swapChainImage.get(), sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost);
    clear = false;
}
