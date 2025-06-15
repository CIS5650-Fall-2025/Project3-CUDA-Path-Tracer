// texture_utils.h
#pragma once
#include "texture.h"
#include <cuda_runtime.h>

bool createCudaTexture(const HostTexture<unsigned char>& hostTex, Texture& gpuTex);
bool createCudaTexture(const HostTexture<float>& hostTex, Texture& gpuTex);
