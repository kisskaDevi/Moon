#ifndef RAYH
#define RAYH
#include "vec4.h"

class ray
{
private:
    vec4 origin;
    vec4 direction;

public:
    __host__ __device__ ray() {}
    __host__ __device__ ray(const vec4& origin, const vec4& direction) : origin(origin), direction(normal(direction)){}
    __host__ __device__ vec4 getOrigin() const { return origin; }
    __host__ __device__ vec4 getDirection() const { return direction; }
    __host__ __device__ vec4 point(const float& t) const { return origin + t * direction; }
};

#endif
