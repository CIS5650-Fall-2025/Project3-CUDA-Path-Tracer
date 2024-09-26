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
__host__ __device__ bool scatterDielectric(
    PathSegment& pathSegment,
    const glm::vec3 intersect,
    const glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    pathSegment.ray.origin = intersect + normal * EPSILON;

    pathSegment.remainingBounces--;

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