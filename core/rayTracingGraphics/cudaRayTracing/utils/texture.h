#ifndef CUDA_TEXTURE_H
#define CUDA_TEXTURE_H

#include <filesystem>
#include <stb_image.h>

#include "utils/buffer.h"

namespace cuda::rayTracing {

struct Texture
{
    std::filesystem::path path;
    Buffer<uint8_t> buffer;
    cudaTextureObject_t object{0};

    Texture() = default;
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&&) = default;
    Texture& operator=(Texture&&) = default;
    Texture(const std::filesystem::path& path) : path(path) {
        create(path);
    }

    void create(const std::filesystem::path& texturePath = std::filesystem::path()){
        path = texturePath.empty() ? path : texturePath;

        int texWidth = 0, texHeight = 0, texChannels = 0;
        uint8_t* datd = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

        buffer = Buffer<uint8_t>(4 * texWidth * texHeight, datd);

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = buffer.get();
        resDesc.res.pitch2D.width = texWidth;
        resDesc.res.pitch2D.height = texHeight;
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
        resDesc.res.pitch2D.pitchInBytes = 4 * texWidth;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.normalizedCoords = true;
        texDesc.readMode = cudaTextureReadMode::cudaReadModeElementType;
        texDesc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
        checkCudaErrors(cudaCreateTextureObject(&object, &resDesc, &texDesc, NULL));
        checkCudaErrors(cudaGetLastError());
    }
};

}
#endif // CUDA_TEXTURE_H
