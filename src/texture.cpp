#include "texture.h"
#include <iostream>
#include <cuda_runtime.h>

TextureInfo uploadTextureToGPU(const TextureData& texData) {
    TextureInfo texInfo;
    texInfo.filepath = texData.filepath;

    // Validate basic texture properties
    if (!texData.pixels) {
        std::cerr << "No pixel data to upload for: " << texData.filepath << std::endl;
        return texInfo;
    }

    if (texData.width <= 0 || texData.height <= 0) {
        std::cerr << "Invalid texture size: " << texData.width << " x " << texData.height << std::endl;
        return texInfo;
    }

    if (texData.channels != 4) {
        std::cerr << "Expected 4-channel RGBA texture. Got " << texData.channels
            << " (original image likely had fewer channels, but stb_image padded to 4).\n"
            << "Proceeding assuming pixel buffer is 4 channels.\n";
    }

    // Allocate CUDA array (RGBA8 format)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray = nullptr;
    cudaError_t err = cudaMallocArray(&cuArray, &channelDesc, texData.width, texData.height);
    if (err != cudaSuccess || cuArray == nullptr) {
        std::cerr << "cudaMallocArray failed: " << cudaGetErrorString(err) << std::endl;
        return texInfo;
    }

    // Copy from host buffer into CUDA array
    size_t rowPitch = texData.width * 4;  // 4 bytes per pixel (RGBA)

    unsigned char* ptr = texData.pixels;
    std::cout << "Pixel pointer: " << reinterpret_cast<void*>(ptr)
        << ", width = " << texData.width
        << ", height = " << texData.height
        << ", rowPitch = " << texData.width * 4 << std::endl;


    err = cudaMemcpy2DToArray(
        cuArray,
        0, 0,
        texData.pixels,
        rowPitch,
        rowPitch,
        texData.height,
        cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy2DToArray failed: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(cuArray);
        return texInfo;
    }

    // Setup resource description
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Setup texture sampling description
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    // Create the texture object
    cudaTextureObject_t texObj = 0;
    err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "cudaCreateTextureObject failed: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(cuArray);
        return texInfo;
    }

    // Store results
    texInfo.cuArray = cuArray;
    texInfo.texObj = texObj;

    std::cout << "Uploaded texture: " << texData.filepath
        << " (" << texData.width << "x" << texData.height << "), texObj = " << texObj << std::endl;

    return texInfo;
}
