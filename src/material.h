#pragma once
#include "utilities.h"
#include <cuda_runtime.h>

enum MaterialType
{
    Lambertian,
    Specular,
    Microfacet,
    MetallicWorkflow,
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

    __device__ float microfacetPDF(const glm::vec3& wo, const glm::vec3& wh);

    __device__ glm::vec3 microfacetSamplWh(const glm::vec3& wo, const glm::vec2& rng);

    __device__ glm::vec3 metallicWorkflowPDF(const glm::vec3& wo, const glm::vec3& wh);

    __device__ glm::vec3 metallicWorkflowSample(const glm::vec3& wo, const glm::vec2& rng);

    __device__ glm::vec3 samplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    __device__ glm::vec3 lambertianSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    __device__ glm::vec3 specularSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    __device__ glm::vec3 microfacetSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    __device__ glm::vec3 metallicWorkflowSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    
};

