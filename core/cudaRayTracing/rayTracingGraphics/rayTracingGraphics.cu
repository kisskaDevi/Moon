#include "rayTracingGraphics.h"
#include "operations.h"
#include "ray.h"
#include "hitableList.h"
#include "material.h"

#include "core/utils/vkdefault.h"
#include "core/utils/operations.h"


__host__ __device__ inline vec4 max(const vec4& v1, const vec4& v2) {
    return vec4(    v1.x() >= v2.x() ? v1.x() : v2.x(),
                v1.y() >= v2.y() ? v1.y() : v2.y(),
                v1.z() >= v2.z() ? v1.z() : v2.z(),
                v1.w() >= v2.w() ? v1.w() : v2.w());
}

__host__ __device__ inline vec4 min(const vec4& v1, const vec4& v2) {
    return vec4(v1.x() < v2.x() ? v1.x() : v2.x(),
                v1.y() < v2.y() ? v1.y() : v2.y(),
                v1.z() < v2.z() ? v1.z() : v2.z(),
                v1.w() < v2.w() ? v1.w() : v2.w());
}


void rayTracingGraphics::createGraphics()
{
    width = this->swapChainKHR->getExtent().width;
    height = this->swapChainKHR->getExtent().height;

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device.getLogical(), &poolInfo, nullptr, &commandPool);

    finalAttachment.create(
        device.instance,
        device.getLogical(),
        swapChainKHR->getFormat(),
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        swapChainKHR->getExtent(),
        swapChainKHR->getImageCount()
    );
    VkSamplerCreateInfo SamplerInfo = vkDefault::samler();
    vkCreateSampler(device.getLogical(), &SamplerInfo, nullptr, &finalAttachment.sampler);

    imageInfo swapChainInfo{
        swapChainKHR->getImageCount(),
        swapChainKHR->getFormat(),
        offset,
        swapChainKHR->getExtent(),
        swapChainKHR->getExtent(),
        VK_SAMPLE_COUNT_1_BIT
    };

    Link.setImageCount(imageCount);
    Link.setDeviceProp(device.getLogical());
    Link.setShadersPath(shadersPath);
    Link.createDescriptorSetLayout();
    Link.createPipeline(&swapChainInfo);
    Link.createDescriptorPool();
    Link.createDescriptorSets();
    Link.updateDescriptorSets(&finalAttachment);

    colorImage = cuda::buffer<vec4>(width * height);
    bloomImage = cuda::buffer<vec4>(width * height);
    swapChainImage  = cuda::buffer<uint32_t>(width * height);

    checkCudaErrors(cudaMalloc((void**)&randState, width * height * sizeof(curandState)));

    Buffer::create(
        device.instance,
        device.getLogical(),
        sizeof(uint32_t) * width * height,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &stagingBuffer.instance,
        &stagingBuffer.memory
    );

    hostFrameBuffer = new uint32_t[width * height];
    vkMapMemory(device.getLogical(), stagingBuffer.memory, 0, sizeof(uint32_t) * width * height, 0, &stagingBuffer.map);
}

void rayTracingGraphics::destroyGraphics() {
    delete[] hostFrameBuffer;

    stagingBuffer.destroy(device.getLogical());
    if(commandPool) {vkDestroyCommandPool(device.getLogical(), commandPool, nullptr); commandPool = VK_NULL_HANDLE;}
    finalAttachment.deleteAttachment(device.getLogical());
    finalAttachment.deleteSampler(device.getLogical());
    Link.destroy();
    checkCudaErrors(cudaFree(randState));
    colorImage.destroy();
    bloomImage.destroy();
    swapChainImage.destroy();
}

__device__ vec4 scatter(const ray& r, const vec4& norm, const properties& props, curandState* local_rand_state)
{
    vec4 scattered = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    if(props.emissionFactor < 0.98f)
    {
        const vec4& d = r.getDirection();

        if (curand_uniform(local_rand_state) <= props.refractProb) {
            const vec4 n = (dot(d, norm) <= 0.0f) ? norm : - norm;
            const float eta = (dot(d, norm) <= 0.0f) ? (1.0f / props.refractiveIndex) : props.refractiveIndex;

            float cosPhi = dot(d, n);
            float sinTheta = eta * std::sqrt(1.0f - cosPhi * cosPhi);
            if (std::abs(sinTheta) <= 1.0f) {
                float cosTheta = std::sqrt(1.0f - sinTheta * sinTheta);
                vec4 tau = normal(d - dot(d, n) * n);
                scattered = sinTheta * tau - cosTheta * n;
            }

            if (scattered.length2() == 0.0f) {
                if (props.fuzz > 0.0f) {
                    vec4 reflect = normal(d + 2.0f * std::abs(dot(d, n)) * n);
                    scattered = reflect + props.fuzz * random_in_unit_sphere(reflect, props.angle, local_rand_state);
                    scattered = (dot(n, scattered) > 0.0f ? 1.0f : 0.0f) * scattered;
                } else {
                    scattered = random_in_unit_sphere(norm, props.angle, local_rand_state);
                    scattered = (dot(n, scattered) > 0.0f ? 1.0f : -1.0f) * scattered;
                }
            }
        } else {
            if (props.fuzz > 0.0f) {
                vec4 reflect = normal(d + 2.0f * std::abs(dot(d, norm)) * norm);
                scattered = reflect + props.fuzz * random_in_unit_sphere(reflect, props.angle, local_rand_state);
                scattered = (dot(norm, scattered) > 0.0f ? 1.0f : 0.0f) * scattered;
            } else {
                scattered = random_in_unit_sphere(norm, props.angle, local_rand_state);
                scattered = (dot(norm, scattered) > 0.0f ? 1.0f : -1.0f) * scattered;
            }
        }
    }

    return scattered;
}

namespace base {
    __device__ vec4 color(ray r, size_t maxIterations, hitableContainer* container, curandState* local_rand_state) {
        vec4 color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
        hitRecord rec;

        for (; maxIterations > 0; maxIterations--) {
            if (r.getDirection().length2() > 0.0f && container->hit(r, 0.001f, FLT_MAX, rec)) {
                color = min(rec.color, color);
                r = ray(rec.point, scatter(r, rec.normal, rec.props, local_rand_state));
            } else {
                break;
            }
        }
        return  rec.props.emissionFactor >= 1.0f
                   ? vec4(color.x(), color.y(), color.z(), 1.0f)
                   : vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

namespace bloom {
    __device__ vec4 color(ray r, size_t maxIterations, hitableContainer* container, curandState* local_rand_state) {
        vec4 color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
        hitRecord rec;
        r = ray(r.getOrigin(), random_in_unit_sphere(r.getDirection(), 0.025 * pi, local_rand_state));

        for (; maxIterations > 0; maxIterations--) {
            if (r.getDirection().length2() > 0.0f && container->hit(r, 0.001f, FLT_MAX, rec)) {
                color = min(rec.color, color);
                r = ray(rec.point, scatter(r, rec.normal, rec.props, local_rand_state));
            } else {
                break;
            }
        }
        return  (rec.props.emissionFactor >= 1.0f) && (color.x() >= 0.9f || color.y() >= 0.9f || color.z() >= 0.9f)
                ? vec4(color.x(), color.y(), color.z(), 1.0f)
                : vec4(0.0f, 0.0f, 0.0f, 1.0f);
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
        dst[pixel] = (uint32_t(255.0f*res[2]) << 0 | uint32_t(255.0f*res[1]) << 8 | uint32_t(255.0f*res[0]) << 16 | uint32_t(255) << 24 );
    }
}

std::vector<std::vector<VkSemaphore>> rayTracingGraphics::submit(const std::vector<std::vector<VkSemaphore>>&, const std::vector<VkFence>&, uint32_t imageIndex)
{
    dim3 blocks(width / xThreads + 1, height / yThreads + 1, 1);
    dim3 threads(xThreads, yThreads, 1);

    render<<<blocks, threads>>>(clear, width, height, rayDepth, randState, swapChainImage.get(), colorImage.get(), bloomImage.get(), cam, container);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    clear = false;

    cudaMemcpy(hostFrameBuffer, swapChainImage.get(), sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost);
    std::memcpy(stagingBuffer.map, hostFrameBuffer, sizeof(uint32_t) * width * height);

    VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
    Texture::transitionLayout(commandBuffer, finalAttachment.instances[imageIndex].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    Texture::copyFromBuffer(commandBuffer, stagingBuffer.instance, finalAttachment.instances[imageIndex].image, {width, height, 1}, 1);
    Texture::transitionLayout(commandBuffer, finalAttachment.instances[imageIndex].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
    SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool, &commandBuffer);

    return std::vector<std::vector<VkSemaphore>>();
}
