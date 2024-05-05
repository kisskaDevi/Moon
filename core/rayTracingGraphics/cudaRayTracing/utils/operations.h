#ifndef CUDA_OPERATIONSH
#define CUDA_OPERATIONSH

#include "math/vec4.h"
#include <string>

namespace cuda::rayTracing {

#define checkCudaErrors(val) debug::check_cuda( (val), #val, __FILE__, __LINE__)

namespace debug {

    void check_cuda(
        cudaError_t             result,
        char const* const       func,
        const char* const       file,
        int const               line);
}

namespace image {

    void outPPM(
        vec4f*                  frameBuffer,
        size_t                  width,
        size_t                  height,
        const std::string&      filename);

    void outPGM(
        vec4f*                  frameBuffer,
        size_t                  width,
        size_t                  height,
        const std::string&      filename);
}

}
#endif
