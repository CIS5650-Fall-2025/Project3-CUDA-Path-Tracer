#pragma once
#include <string>
#include <cuda_runtime.h>
#include <stb_image.h>


struct TextureData {
    std::string filepath;
    unsigned char* pixels = nullptr;  // CPU-side buffer
    int width = 0;
    int height = 0;
    int channels = 0;

    ~TextureData() {
        if (pixels) {
            stbi_image_free(pixels);
            pixels = nullptr;
        }
    }
};


struct TextureInfo {
    std::string filepath;
    cudaArray_t cuArray = nullptr;
    cudaTextureObject_t texObj = 0;
};


TextureInfo uploadTextureToGPU(const TextureData& texData);