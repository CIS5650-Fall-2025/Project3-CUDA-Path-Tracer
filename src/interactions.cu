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
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    thrust::uniform_real_distribution<float> u01(0, 1);
    float shadeSampler = u01(rng);
    glm::vec3 newDirection;
    glm::vec3 inDirection = normalize(pathSegment.ray.direction);
    glm::vec3 surfaceNormal = normalize(normal);

    if (shadeSampler < m.hasRefractive) {
        // angle between incoming ray and surface normal
        float cosTheta = -glm::dot(normal, pathSegment.ray.direction);
        float eta = (cosTheta > 0) ? (1 / m.indexOfRefraction) : (m.indexOfRefraction);
        glm::vec3 refractDirection = glm::normalize(
            glm::refract(inDirection, surfaceNormal, eta));
        float r0 = pow((1.0f - eta) / (1.0f + eta), 2);
        float schlickFactor = r0 + (1.0f - r0) * pow(1.0f - cosTheta, 5);

        if (glm::length(refractDirection) == 0 || u01(rng) < schlickFactor) {
            pathSegment.ray.direction = glm::reflect(inDirection, surfaceNormal);
        }
        else {
            pathSegment.ray.direction = refractDirection;
        }

    }
    else if (shadeSampler < m.hasReflective) {
        pathSegment.ray.direction = glm::reflect(inDirection, surfaceNormal);
    }
    else {
        pathSegment.ray.direction = normalize(calculateRandomDirectionInHemisphere(surfaceNormal, rng));
    }
    pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
}
