#pragma once

#include <string>
#include <cuda_runtime.h>

// Generic texture wrapper
template<typename T>
struct HostTexture {
    int width = 0;
    int height = 0;
    int channels = 0;
    T* data = nullptr; // Host memory
};

// Loads either 8-bit or HDR float texture into HostTexture<T>
template<typename T>
bool loadTexture(const std::string& filename, HostTexture<T>& outTex);

// Frees allocated host memory
template<typename T>
void freeTexture(HostTexture<T>& tex);

// CUDA handle struct
struct Texture {
    cudaTextureObject_t texObj = 0;
    cudaArray_t cuArray = nullptr;
};
