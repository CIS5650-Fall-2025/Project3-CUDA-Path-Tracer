#include "texture_utils.h"
#include <cuda_runtime.h>
#include <iostream>

// ------------------------------------------------------------------------------------------------
// Uploads an 8-bit RGBA texture (HostTexture<unsigned char>) to a CUDA texture object
// ------------------------------------------------------------------------------------------------
bool createCudaTexture(const HostTexture<unsigned char>& hostTexture, Texture& gpuTexture) {
    if (!hostTexture.data) {
        gpuTexture.texObj = 0;
        gpuTexture.cuArray = nullptr;
        return true;
    }

    const int width = hostTexture.width;
    const int height = hostTexture.height;
    const size_t rowBytes = width * 4 * sizeof(unsigned char); // RGBA8

    // Create 8-bit channel descriptor
    const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    // Allocate CUDA array
    cudaError_t cudaStatus = cudaMallocArray(&gpuTexture.cuArray, &channelDesc, width, height);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "[createCudaTexture] cudaMallocArray failed: "
            << cudaGetErrorString(cudaStatus) << "\n";
        return false;
    }

    // Copy pixel data to CUDA array
    cudaStatus = cudaMemcpy2DToArray(
        gpuTexture.cuArray, 0, 0,
        hostTexture.data,
        rowBytes,
        rowBytes,
        height,
        cudaMemcpyHostToDevice
    );
    if (cudaStatus != cudaSuccess) {
        std::cerr << "[createCudaTexture] cudaMemcpy2DToArray failed: "
            << cudaGetErrorString(cudaStatus) << "\n";
        cudaFreeArray(gpuTexture.cuArray);
        gpuTexture.cuArray = nullptr;
        return false;
    }

    // Describe the CUDA array as a resource
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = gpuTexture.cuArray;

    // Configure texture sampling
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat; // Converts [0,255] to [0.0,1.0]
    texDesc.normalizedCoords = 1;

    // Create the CUDA texture object
    cudaStatus = cudaCreateTextureObject(&gpuTexture.texObj, &resDesc, &texDesc, nullptr);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "[createCudaTexture] cudaCreateTextureObject failed: "
            << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFreeArray(gpuTexture.cuArray);
        gpuTexture.cuArray = nullptr;
        return false;
    }

    return true;
}

// ------------------------------------------------------------------------------------------------
// Uploads an HDR float RGBA texture (HostTexture<float>) to a CUDA texture object
// ------------------------------------------------------------------------------------------------
bool createCudaTexture(const HostTexture<float>& hostTexture, Texture& gpuTexture) {
    if (!hostTexture.data) {
        gpuTexture.texObj = 0;
        gpuTexture.cuArray = nullptr;
        return true;
    }

    const int width = hostTexture.width;
    const int height = hostTexture.height;
    const size_t rowBytes = width * 4 * sizeof(float); // RGBA32F

    // Create float4 channel descriptor
    const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    // Allocate CUDA array
    cudaError_t cudaStatus = cudaMallocArray(&gpuTexture.cuArray, &channelDesc, width, height);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "[createCudaTexture HDR] cudaMallocArray failed: "
            << cudaGetErrorString(cudaStatus) << std::endl;
        return false;
    }

    // Copy pixel data to CUDA array
    cudaStatus = cudaMemcpy2DToArray(
        gpuTexture.cuArray, 0, 0,
        hostTexture.data,
        rowBytes,
        rowBytes,
        height,
        cudaMemcpyHostToDevice
    );
    if (cudaStatus != cudaSuccess) {
        std::cerr << "[createCudaTexture HDR] cudaMemcpy2DToArray failed: "
            << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFreeArray(gpuTexture.cuArray);
        gpuTexture.cuArray = nullptr;
        return false;
    }

    // Describe the CUDA array as a resource
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = gpuTexture.cuArray;

    // Configure texture sampling
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType; // Keeps float precision
    texDesc.normalizedCoords = 1;

    // Create the CUDA texture object
    cudaStatus = cudaCreateTextureObject(&gpuTexture.texObj, &resDesc, &texDesc, nullptr);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "[createCudaTexture HDR] cudaCreateTextureObject failed: "
            << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFreeArray(gpuTexture.cuArray);
        gpuTexture.cuArray = nullptr;
        return false;
    }

    return true;
}
