#pragma once

#include <glm/glm.hpp>
#include <thrust/random.h>

__host__ __device__ glm::vec3 randomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng);

__host__ __device__ glm::vec3 randomOnUnitSphere(thrust::default_random_engine& rng);

__host__ __device__ glm::vec2 randomOnUnitCircle(thrust::default_random_engine& rng);