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
    glm::vec3 albedo = glm::vec3(0.5f);
    MaterialType type = MaterialType::Lambertian;
    float metallic = 0.f;
    float roughness = 1.f;
    float ior = 0.f;
    float emittance = 0.f;

    cudaTextureObject_t albedoMap = 0;
    cudaTextureObject_t normalMap = 0;
    cudaTextureObject_t metallicRoughnessMap = 0;

    __device__ float microfacetPDF(const glm::vec3& wo, const glm::vec3& wh);

    __device__ glm::vec3 microfacetSamplWh(const glm::vec3& wo, const glm::vec2& rng);

    __device__ float metallicWorkflowPDF(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& wh);

    __device__ glm::vec3 metallicWorkflowSample(const glm::vec3& wo, const glm::vec3& rng);

    __device__ glm::vec3 metallicWorkflowEval(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& wh);

    __device__ glm::vec3 microfacetEval(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& wh);

    __device__ glm::vec3 samplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    __device__ glm::vec3 lambertianSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    __device__ glm::vec3 specularSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    __device__ glm::vec3 microfacetSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    __device__ glm::vec3 metallicWorkflowSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf);

    __device__ float getPDF(const glm::vec3& nor, glm::vec3 wo, glm::vec3 wi);

    __device__ glm::vec3 getBSDF(const glm::vec3& nor, glm::vec3 wo, glm::vec3 wi, float* pdf);

    __device__ void createMaterialInst(const Material& mat, const glm::vec2& uv);
    
};

