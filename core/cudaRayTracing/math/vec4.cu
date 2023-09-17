#include "vec4.h"

__host__ __device__ vec4 normal(const vec4& v) {
    return v / v.length();
}

__host__ __device__ vec4 cross(const vec4& v1, const vec4& v2) {
    return vec4(v1.y() * v2.z() - v1.z() * v2.y(), v1.z() * v2.x() - v1.x() * v2.z(), v1.x() * v2.y() - v1.y() * v2.x(), 0.0f);
}

__device__ vec4 random_in_unit_sphere(const vec4& direction, const float& angle, curandState* local_rand_state) {
    float phi = 2 * pi * curand_uniform(local_rand_state);
    float theta = angle * curand_uniform(local_rand_state);

    float x = std::sin(theta) * std::cos(phi);
    float y = std::sin(theta) * std::sin(phi);
    float z = std::cos(theta);

    return normal(x * vec4::getHorizontal(direction) + y * vec4::getVertical(direction) + z * direction);
}
