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
    // Pre-fetch
    const glm::vec3 direction = glm::normalize(pathSegment.ray.direction);
    glm::vec3 norm = glm::normalize(normal);

    pathSegment.ray.origin = intersect;
    const glm::vec3 delta = 0.001f * norm;
    pathSegment.color *= m.color;
    --pathSegment.remainingBounces;

    // Diffuse for any material
    thrust::uniform_real_distribution<float> u01(0, 1);
    float r01;

    // Specular reflection
    if (m.hasReflective > 0)
    {
        if (m.roughness <= EPSILON)
        {
            pathSegment.ray.direction = glm::reflect(direction, norm);
        }
        else
        {
            r01 = u01(rng);
            glm::vec3 randomDirectionDelta{ glm::normalize(calculateRandomDirectionInHemisphere(norm, rng)) };
            pathSegment.ray.direction = (1 - m.roughness) * glm::normalize(glm::reflect(direction, norm))
                + m.roughness * randomDirectionDelta;
        }
        pathSegment.ray.origin += delta;
    }
    // Refractive
    else if (m.hasRefractive > 0)
    {
        r01 = u01(rng);

        // Derive reflection coeff R_theta
        const float cos_theta = -glm::dot(norm, direction);
        constexpr float n_i = 1.0f;
        const float n_o = m.indexOfRefraction;
        const float R_0 = glm::pow((n_i - n_o) / (n_i + n_o), 2.0f);
        const float R_theta = R_0 + (1.0f - R_0) * glm::pow(1.0f - cos_theta, 5.0f);
        float dot_n_d = glm::dot(norm, direction);

        if (r01 > R_theta)
        {
            // Refract
            float ratio;
            if (dot_n_d > 0.f)  // material -> air
            {
                norm = -norm;
                ratio = m.indexOfRefraction;
            }
            else  // air -> material
            {
                ratio = 1.0f / m.indexOfRefraction;
            }
            pathSegment.ray.direction = glm::refract(direction, norm, ratio);

            pathSegment.ray.origin -= delta;
        }
        else
        {
            // Reflect
            pathSegment.ray.direction = glm::reflect(direction, norm);
            pathSegment.ray.origin += delta;
        }

        if (m.roughness > EPSILON)
        {
            if ((r01 > R_theta && dot_n_d < 0.f) || (r01 < R_theta && dot_n_d > 0.f)) // BRDF and BTDF
            {
                r01 = u01(rng);
                glm::vec3 randomDirectionDelta{ glm::normalize(calculateRandomDirectionInHemisphere(norm, rng)) };
                pathSegment.ray.direction = (1 - m.roughness) * pathSegment.ray.direction + m.roughness * randomDirectionDelta;
            }
        }
    }
    // Diffuse material
    else
    {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(norm, rng);
        pathSegment.ray.origin += delta;
    }
}
