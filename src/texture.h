#pragma once
//#include <cuda_runtime.h>  
#include <texture_types.h>
#include "utilities.h"

class Texture {
public:
    Texture(const char* filename);
    ~Texture();

    cudaTextureObject_t getTextureObject() const { return texObj; }
	bool createTextureFromFile(const char* filename, cudaTextureObject_t& texObj, cudaArray_t& cuArray);
	//__device__ float4 get(float u, float v);
private:
    cudaTextureObject_t texObj;
    cudaArray_t cuArray;  

    // 禁止拷贝和赋值
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
};