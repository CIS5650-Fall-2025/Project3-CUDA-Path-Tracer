#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ float TrowbridgeReitzPdf(glm::vec3 wo, glm::vec3 wh, float roughness);
__device__ glm::vec3 Sample_wh(glm::vec3 wo, glm::vec2 xi, float roughness);
__device__ bool SameHemisphere(glm::vec3 w, glm::vec3 wp);
__device__ glm::vec3 f_microfacet_refl(glm::vec3 albedo, glm::vec3 wo, glm::vec3 wi, float roughness);