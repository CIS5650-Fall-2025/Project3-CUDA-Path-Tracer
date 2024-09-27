#pragma once

#include "sceneStructs.h"
#include "samplers.h"
#include "utilities.h"
#include <glm/glm.hpp>
#include <thrust/random.h>

// Lambertian
#pragma region
__host__ __device__ bool scatterLambertian(
    PathSegment& pathSegment,
    const glm::vec3 intersect,
    const glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    pathSegment.ray.origin = intersect + normal * EPSILON;
    pathSegment.ray.direction = randomDirectionInHemisphere(normal, rng);

    pathSegment.remainingBounces--;

    return true;
}

__host__ __device__ glm::vec3 evalLambertian(
    const glm::vec3& dirIn,
    const glm::vec3& dirOut,
    const glm::vec3& normal,
    const Material& m)
{
    return m.color * max(0.0f, glm::dot(dirOut, normal) / PI);
}

__host__ __device__ float pdfLambertian(
    const glm::vec3& dirIn,
    const glm::vec3& dirOut,
    const glm::vec3& normal)
{
    return max(0.0f, glm::dot(dirOut, normal) / PI);
}
#pragma endregion

// Metal
#pragma region
__host__ __device__ bool scatterMetal(
    PathSegment& pathSegment,
    const glm::vec3 intersect,
    const glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    pathSegment.ray.origin = intersect + normal * EPSILON;

    // Imperfect specular lighting based on
    // https://raytracing.github.io/books/RayTracingInOneWeekend.html#metal/fuzzyreflection
    // i.e. we importance sample a point on a unit sphere 
    // (uniformly w.r.t. surface area), scale it by roughness, 
    // and tweak ray direction by this offset
    glm::vec3 reflected = glm::reflect(glm::normalize(pathSegment.ray.direction), normal);
    reflected += randomOnUnitSphere(rng) * m.roughness;
    reflected = glm::normalize(reflected);

    pathSegment.ray.direction = reflected;

    pathSegment.remainingBounces--;

    return (glm::dot(reflected, normal) > 0);
}

__host__ __device__ glm::vec3 evalMetal(
    const glm::vec3& dirIn,
    const glm::vec3& dirOut,
    const glm::vec3& normal,
    const Material& m)
{
    return m.color;
}

__host__ __device__ float pdfMetal(
    const glm::vec3& dirIn,
    const glm::vec3& dirOut,
    const glm::vec3& normal)
{
    return 0.0f;
}
#pragma endregion

// Dielectric
#pragma region
__host__ __device__ bool refract(const glm::vec3& v, const glm::vec3& n, float iorIOverT, glm::vec3& refracted, float& cosTheta2)
{
    glm::vec3 uv = glm::normalize(v);
    float dt = glm::dot(uv, n);
    float discrim = 1.0f - iorIOverT * iorIOverT * (1.0f - dt * dt);
    if (discrim)
    {
        cosTheta2 = std::sqrt(discrim);
        refracted = iorIOverT * (uv - n * dt) - n * cosTheta2;
        return true;
    }
    return false;
}

__host__ __device__ bool scatterDielectric(
    PathSegment& pathSegment,
    const glm::vec3 intersect,
    const glm::vec3 normal_,
    const Material& m,
    thrust::default_random_engine& rng)
{
    float eta1, eta2;
    glm::vec3 normal;

    // Ensure ior and normal are correctly oriented for computing reflection and refraction
    // Assume we are going from "air" of ior 1 into a medium of ior indexOfRefraction
    if (glm::dot(pathSegment.ray.direction, normal_) > 0.0f)
    {
        normal = -normal_;
        eta1 = m.indexOfRefraction;
        eta2 = 1.0f;
    }
    else
    {
        normal = normal_;
        eta1 = 1.0f;
        eta2 = m.indexOfRefraction;
    }

    // Compute reflected + refracted ray
    float cosTheta2, cosTheta1 = glm::dot(pathSegment.ray.direction, -normal) / glm::length(pathSegment.ray.direction);
    glm::vec3 refracted, reflected = glm::reflect(pathSegment.ray.direction, normal_);

    pathSegment.remainingBounces--;

    if (!refract(pathSegment.ray.direction, normal, eta1 / eta2, refracted, cosTheta2))
    {
        // No refraction, only reflection
        pathSegment.ray.direction = reflected;
        pathSegment.ray.origin = intersect + reflected * 0.01f;
        return true;
    }

    // Compute fresnel coefficient
    float rhoParallel = ((eta2 * cosTheta1) - (eta1 * cosTheta2)) / ((eta2 * cosTheta1) + (eta1 * cosTheta2));
    float rhoPerp = ((eta1 * cosTheta1) - (eta2 * cosTheta2)) / ((eta1 * cosTheta1) + (eta2 * cosTheta2));
    float fresnel = (rhoParallel * rhoParallel + rhoPerp * rhoPerp) * 0.5f;

    // Sample scattered or reflected ray
    thrust::uniform_real_distribution<float> u01(0, 1);
    if (u01(rng) < fresnel) {
        // Reflect
        pathSegment.ray.direction = reflected;
        pathSegment.ray.origin = intersect + reflected * 0.01f;
    } else {
        // Refract
        pathSegment.ray.direction = refracted;
        pathSegment.ray.origin = intersect + refracted * 0.01f;
    }

    return true;
}

__host__ __device__ glm::vec3 evalDielectric(
    const glm::vec3& dirIn,
    const glm::vec3& dirOut,
    const glm::vec3& normal,
    const Material& m)
{
    return m.color;
}

__host__ __device__ float pdfDielectric(
    const glm::vec3& dirIn,
    const glm::vec3& dirOut,
    const glm::vec3& normal)
{
    return 1.0f;
}
#pragma endregion

// Emissive
#pragma region
__host__ __device__ bool scatterEmissive(
    PathSegment& pathSegment,
    const glm::vec3 intersect,
    const glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    pathSegment.remainingBounces = -1;

    return true;
}

__host__ __device__ glm::vec3 evalEmissive(
    const glm::vec3& dirIn,
    const glm::vec3& dirOut,
    const glm::vec3& normal,
    const Material& m)
{
    return m.color * m.emittance;
}

__host__ __device__ float pdfEmissive(
    const glm::vec3& dirIn,
    const glm::vec3& dirOut,
    const glm::vec3& normal)
{
    return 1.0f;
}
#pragma endregion