#include "operations.h"

#include <fstream>
#include <iostream>

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

namespace cuda::Buffer {
    void* Buffer::create(size_t size) {
        void* buffer;
        checkCudaErrors(cudaMalloc((void**)&buffer, size));
        return buffer;
    }
}

namespace cuda::Image {
    void Image::outPPM(vec4* frameBuffer, size_t width, size_t height, const std::string& filename) {
        vec4* hostFrameBuffer = new vec4[width * height];
        cudaMemcpy(hostFrameBuffer, frameBuffer, width * height * sizeof(vec4), cudaMemcpyDeviceToHost);

        std::ofstream image(filename);
        image << "P3\n" << width << " " << height << "\n255\n";
        for (size_t j = 0; j < height; j++) {
            for (size_t i = 0; i < width; i++) {
                size_t pixel_index = j * width + (width - 1 - i);
                image   << static_cast<uint32_t>(255.99f * hostFrameBuffer[pixel_index].r()) << " "
                        << static_cast<uint32_t>(255.99f * hostFrameBuffer[pixel_index].g()) << " "
                        << static_cast<uint32_t>(255.99f * hostFrameBuffer[pixel_index].b()) << "\n";
            }
        }
        image.close();
        delete[] hostFrameBuffer;
    }

    void Image::outPGM(vec4* frameBuffer, size_t width, size_t height, const std::string& filename) {
        vec4* hostFrameBuffer = new vec4[width * height];
        cudaMemcpy(hostFrameBuffer, frameBuffer, width * height * sizeof(vec4), cudaMemcpyDeviceToHost);

        std::ofstream image(filename);
        image << "P2\n" << width << " " << height << "\n255\n";
        for (size_t j = 0; j < height; j++) {
            for (size_t i = 0; i < width; i++) {
                size_t pixel_index = j * width + (width - 1 - i);
                image << static_cast<uint32_t>(255.99f * (hostFrameBuffer[pixel_index].r() + hostFrameBuffer[pixel_index].g() + hostFrameBuffer[pixel_index].b()) / 3) << "\n";
            }
        }
        image.close();
        delete[] hostFrameBuffer;
    }
}
