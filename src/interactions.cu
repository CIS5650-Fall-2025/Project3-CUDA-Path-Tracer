#include "interactions.h"

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

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)

{
    if (pathSegment.remainingBounces == 0)
        return;

    normal = glm::normalize(normal);

    glm::vec3 newDirection;
    // Generate a random number to decide between diffuse, reflective and refractive
    thrust::uniform_real_distribution<float> u01(0, 1);
    float rand = u01(rng);

    if (rand < m.hasReflective) {
        // Reflective (specular) surface
        newDirection = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color;
    }
    else if (rand < m.hasReflective + m.hasRefractive) {
        // Refractive surface
        float n1 = 1.0f; // Assume air as the surrounding medium
        float n2 = m.indexOfRefraction;
        float eta = n1 / n2;
        float cosThetaI = glm::dot(-pathSegment.ray.direction, normal);
        float k = 1.0f - eta * eta * (1.0f - cosThetaI * cosThetaI);

        if (k >= 0.0f) { // Check for total internal reflection
            newDirection = eta * pathSegment.ray.direction + (eta * cosThetaI - sqrtf(k)) * normal;
            pathSegment.color *= m.color; // You might want to adjust color for refraction
        }
        else {
            // Total internal reflection - treat as reflective
            newDirection = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.color *= m.specular.color;
        }
    }
    else {
        // Diffuse surface
        newDirection = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color;
    }

    pathSegment.ray.origin = intersect + newDirection * 0.001f; // Small offset to avoid self-intersection
    pathSegment.ray.direction = glm::normalize(newDirection);
    pathSegment.remainingBounces--;
}