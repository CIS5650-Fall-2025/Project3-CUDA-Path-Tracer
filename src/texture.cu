#include "texture.h"
#include <cuda_runtime.h>
Texture::Texture(const char* filename) {
	// 创建纹理对象
	if (!createTextureFromFile(filename, texObj, cuArray)) {
		// 创建失败，释放资源
		texObj = 0;
		cuArray = nullptr;
	}
}

Texture::~Texture() {
	// 释放资源
	if (texObj) {
		cudaDestroyTextureObject(texObj);
		cudaFreeArray(cuArray);
	}
}

// 定义函数
bool Texture::createTextureFromFile(const char* filename, cudaTextureObject_t& texObj, cudaArray_t& cuArray) {
    // load image data
    int width, height, channels;
	int desired_channels = 0;  // keep the original number of channels
    unsigned char* h_data8 = nullptr;
    float* h_data32 = nullptr;
    bool isHDR = false;

    if (stbi_is_hdr(filename)) {
		// if file is HDR, load as float
        isHDR = true;
        h_data32 = stbi_loadf(filename, &width, &height, &channels, desired_channels);
        if (!h_data32) {
			printf("Failed to load image: %s\n", filename);
            return false;
        }
    }
    else {
		// if file is LDR, load as unsigned char
        h_data8 = stbi_load(filename, &width, &height, &channels, desired_channels);
        if (!h_data8) {
			printf("Failed to load image: %s\n", filename);
            return false;
        }
    }

	// determine number of channels
    if (channels != 3 && channels != 4) {
		printf("Image must have 3 or 4 channels\n");
        if (isHDR)
            stbi_image_free(h_data32);
        else
            stbi_image_free(h_data8);
        return false;
    }

	// initialize channel descriptor
    cudaChannelFormatDesc channelDesc;
    if (isHDR) {
        if (channels == 3) {
            channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        }
        else {
            channelDesc = cudaCreateChannelDesc<float4>();
        }
    }
    else {
        if (channels == 3) {
            channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        }
        else {
            channelDesc = cudaCreateChannelDesc<uchar4>();
        }
    }

    cudaMallocArray(&cuArray, &channelDesc, width, height);

    if (isHDR) {
        size_t numPixels = width * height;
        float4* h_data4 = new float4[numPixels];

        if (channels == 3) {
            for (size_t i = 0; i < numPixels; ++i) {
                h_data4[i].x = h_data32[i * 3 + 0];
                h_data4[i].y = h_data32[i * 3 + 1];
                h_data4[i].z = h_data32[i * 3 + 2];
                h_data4[i].w = 1.0f;
            }
        }
        else {
            memcpy(h_data4, h_data32, numPixels * sizeof(float4));
        }

        size_t sizeofRow = width * sizeof(float4);
        cudaMemcpy2DToArray(cuArray, 0, 0, h_data4, sizeofRow, sizeofRow, height, cudaMemcpyHostToDevice);

        delete[] h_data4;
        stbi_image_free(h_data32);

    }
    else {
        size_t numPixels = width * height;
        uchar4* h_data4 = new uchar4[numPixels];

        if (channels == 3) {
            for (size_t i = 0; i < numPixels; ++i) {
                h_data4[i].x = h_data8[i * 3 + 0];
                h_data4[i].y = h_data8[i * 3 + 1];
                h_data4[i].z = h_data8[i * 3 + 2];
                h_data4[i].w = 255;
            }
        }
        else {
            memcpy(h_data4, h_data8, numPixels * sizeof(uchar4));
        }

        size_t sizeofRow = width * sizeof(uchar4);
        cudaMemcpy2DToArray(cuArray, 0, 0, h_data4, sizeofRow, sizeofRow, height, cudaMemcpyHostToDevice);

        delete[] h_data4;
        stbi_image_free(h_data8);
    }

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
	texDesc.addressMode[0] = cudaAddressModeWrap;  // or cudaAddressModeClamp
    texDesc.addressMode[1] = cudaAddressModeWrap;     
    texDesc.filterMode = cudaFilterModeLinear;     // or cudaFilterModePoint
    texDesc.readMode = isHDR ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0;

    // 7. 创建纹理对象
    cudaError_t err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
		printf("Failed to create texture object: %s\n", cudaGetErrorString(err));
        cudaFreeArray(cuArray);
        return false;
    }
    // 返回纹理对象和 CUDA 数组
    return true;
}

// __device__ float4 Texture::get(float u, float v) {
//	// 使用纹理对象进行采样
//	return tex2D<float4>(texObj, u, v);
//}