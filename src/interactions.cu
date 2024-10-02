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
    //Meeting light
    if (m.hasReflective > 0.0f && m.hasRefractive <= 0.f) {
        pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
        pathSegment.color *= m.color;
    }
    else if(m.hasReflective <= 0.f && m.hasRefractive <= 0.f){
        glm::vec3 dir = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        pathSegment.ray.direction = glm::normalize(dir);
        pathSegment.color *= m.color;
    }
    else{
        glm::vec3 rayDir = pathSegment.ray.direction;
        float cosTheta = glm::dot(rayDir, normal);
        float n1 = 1.0f;
        float n2 = 1.0f;

        if (cosTheta < 0.f)
        {
            n2 = m.indexOfRefraction;
        }
        else
        {
            normal = glm::normalize(-normal);
            cosTheta = glm::dot(rayDir, normal);
            n1 = m.indexOfRefraction;
        }

        float R0 = glm::pow((n1 - n2) / (n1 + n2), 2);
        float R = R0 + (1 - R0) * glm::pow((1 - cosTheta), 5);

        thrust::uniform_real_distribution<float> u01(0, 1);
        if (u01(rng) >= R)
        {
            glm::vec3 direction = glm::normalize(glm::refract(rayDir, normal, n2 / n1));
            pathSegment.ray.direction = direction;
            pathSegment.color *= m.color;
        }
        else
        {
            glm::vec3 direction = glm::reflect(rayDir, normal);
            pathSegment.ray.direction = glm::normalize(direction);
            pathSegment.color *= m.color;
        }
    }
    pathSegment.ray.origin = intersect + EPSILON * normal;
}
