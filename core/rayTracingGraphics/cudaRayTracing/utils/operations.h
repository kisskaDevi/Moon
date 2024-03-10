#ifndef CUDA_OPERATIONSH
#define CUDA_OPERATIONSH

#include "math/vec4.h"
#include <string>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

namespace cuda::Buffer {
    void* create(
        size_t                  size);
}

namespace cuda::Image{
    void outPPM(
        vec4*                   frameBuffer,
        size_t                  width,
        size_t                  height,
        const std::string&      filename);

    void outPGM(
        vec4*                   frameBuffer,
        size_t                  width,
        size_t                  height,
        const std::string&      filename);
}

#endif
