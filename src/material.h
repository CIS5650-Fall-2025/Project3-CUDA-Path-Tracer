#pragma once
#include "utilities.h"
#include <cuda_runtime.h>

enum MaterialType
{
    Lambertian,
    Pbr,
    Light
};



struct Material
{

    MaterialType type = MaterialType::Lambertian;
    glm::vec3 albedo = glm::vec3(0.5f);
    float metallic = 0.f;
    float roughness = 1.f;
    float ior = 1.f;
    float emittance = 0.f;

    __device__ glm::vec3 samplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    __device__ glm::vec3 lambertianSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    
};

