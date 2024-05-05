#ifndef CAMERAH
#define CAMERAH

#include "math/ray.h"

namespace cuda::rayTracing {

struct Camera {
    ray viewRay;
    vec4f horizontal;
    vec4f vertical;

    float aspect{ 1.0f };
    float matrixScale{ 0.04f };
    float matrixOffset{ 0.05f };
    float focus{ 0.049f };
    float apertura{ 0.005f };

    __host__ __device__ void update();

    __host__ __device__ Camera();

    __host__ __device__ Camera(
        const ray viewRay,
        float aspect,
        float matrixScale,
        float matrixOffset,
        float focus,
        float apertura);

    __host__ __device__ Camera(const ray& viewRay, float aspect);

    __device__ ray getPixelRay(float u, float v, curandState* local_rand_state) const ;
    __host__ __device__ ray getPixelRay(float u, float v) const ;
};

}
#endif
