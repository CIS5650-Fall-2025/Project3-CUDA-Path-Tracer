#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <thrust/random.h>

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 randomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng);

__host__ __device__ glm::vec3 randomOnUnitSphere(thrust::default_random_engine& rng);

__host__ __device__ glm::vec2 randomOnUnitCircle(thrust::default_random_engine& rng);

__host__ __device__ glm::vec4 sampleBilinear(ImageTextureInfo imageTextureInfo, glm::vec2 uv, glm::vec4* textures);

__host__ __device__ glm::vec4 sampleEnvironmentMap(ImageTextureInfo imageTextureInfo, glm::vec3 rayDir, glm::vec4* textures);