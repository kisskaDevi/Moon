#ifndef MATERIALH
#define MATERIALH

#include "math/ray.h"

struct properties
{
    float refractiveIndex{ 1.0f };
    float refractProb{ 0.0f };
    float fuzz{ 0.0f };
    float angle{ 0.0f };
    float emissionFactor{ 0.0f };
};

__device__ inline vec4 scatter(const ray& r, const vec4& norm, const properties& props, curandState* local_rand_state)
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

#endif
