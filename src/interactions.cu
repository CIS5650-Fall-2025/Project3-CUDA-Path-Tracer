#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void diffuseBRDF(PathSegment& pathSegment, glm::vec3 normal,
    const Material &m, thrust::default_random_engine &rng) {
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.color *= m.color;
}

__host__ __device__ void specularBRDF(PathSegment& pathSegment, glm::vec3 normal, const Material& m) {
    pathSegment.ray.direction = glm::reflect(glm::normalize(pathSegment.ray.direction), normal);
    pathSegment.color *= m.color;
}

__host__ __device__ float fresnel(float cos, float n1, float n2) {
    float sqrtR0 = (n1 - n2) / (n1 + n2);
    float R0 = sqrtR0 * sqrtR0;
    float F = R0 + (1.0f - R0) * powf(1.0f - cos, 5.0f);
    return F;
}

__host__ __device__ void transmissiveBRDF(PathSegment& pathSegment, glm::vec3 normal, 
    const Material& m, thrust::default_random_engine& rng) {
    glm::vec3 inDir = glm::normalize(pathSegment.ray.direction);

    bool inside = glm::dot(inDir, normal) > 0.0f;
    float inEta = inside ? m.indexOfRefraction : 1.0f;
    float outEta = inside ? 1.0f : m.indexOfRefraction;
    float eta = inEta / outEta;

    glm::vec3 N = inside ? -normal : normal;
    
    float cosThetaI = glm::clamp(-glm::dot(inDir, N), 0.f, 1.f);

    glm::vec3 reflectDir = glm::reflect(inDir, N);
    glm::vec3 refractDir = glm::refract(inDir, N, eta);

    float f = fresnel(cosThetaI, inEta, outEta);
    thrust::uniform_real_distribution<float> u01(0, 1);

    pathSegment.ray.direction = u01(rng) < f || glm::length(refractDir) < 1e-10f ? 
        reflectDir : refractDir;
    pathSegment.color *= m.color;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    if (m.hasReflective > 0.0f && m.hasReflective < 1.0f) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        if (u01(rng) < 1.0 - m.hasReflective) {
            diffuseBRDF(pathSegment, normal, m, rng);
            pathSegment.color *= (1.0f / (1.0f - m.hasReflective));
        }
        else {
            specularBRDF(pathSegment, normal, m);
            pathSegment.color *= (1.0f / m.hasReflective);
        }
    }
    else if (m.hasReflective > 0.0f) {
        specularBRDF(pathSegment, normal, m);
    }
    else if (m.hasRefractive > 0.0f) {
        transmissiveBRDF(pathSegment, normal, m, rng);
    }
    else {
        diffuseBRDF(pathSegment, normal, m, rng);
    }

    pathSegment.ray.origin = intersect + 0.01f * pathSegment.ray.direction;
}
