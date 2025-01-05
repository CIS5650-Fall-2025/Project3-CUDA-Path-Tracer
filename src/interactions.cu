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
    glm::vec3 color,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // Calculate the new origin close to the point of intersection
    // so we can bounce the ray. EPSILON = 0.00001f
    pathSegment.ray.origin = intersect + normal * EPSILON;
    
    //===================================================================================
    // PERFECTLY REFLECTIVE MATERIALS
    //===================================================================================
    if (m.hasReflective == 1.0f)
    {
        // Calculate the reflected ray's direction
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.color;
    }
    //===================================================================================
    // PARTIALLY REFLECTIVE MATERIALS
    //===================================================================================
    else if (m.hasReflective > 0.0f)
    {
        // Need to calculate the imperfectly reflected ray direction
        const glm::vec3 reflection = glm::reflect(pathSegment.ray.direction, normal);
        const glm::vec3 direction = glm::normalize(glm::mix(normal, reflection, m.hasReflective));
        const glm::vec3 randomDir = calculateRandomDirectionInHemisphere(direction, rng);
        pathSegment.ray.direction = glm::normalize(
            glm::mix(
                randomDir,
                direction, 
                m.hasReflective)
        );

        // Set the output color
        pathSegment.color *= m.color;
    }
    //===================================================================================
    // REFRACTIVE MATERIALS
    //===================================================================================
    else if (m.hasRefractive == 1.0f) {
        // Normalize the ray direction and the normal vector
        glm::vec3 newRayDirection = glm::normalize(pathSegment.ray.direction);
        glm::vec3 newRayNormal = glm::normalize(normal);

        // Generate a random number between 0 and 1
        thrust::uniform_real_distribution<float> u01(0, 1);
        float randomNum = u01(rng);

        // Compute the Fresnel factor using Schlick's approximation
        const float cosTheta = glm::dot(newRayNormal, newRayDirection);
        const float n_1 = 1.0f;
        const float n_2 = m.indexOfRefraction;
        const float r_0 = glm::pow((n_1 - n_2) / (n_1 + n_2), 2.0f);
        const float fresnelFactor = r_0 + (1.0f - r_0) * glm::pow(1.0f + cosTheta, 5.0f);

        // Determine whether to reflect or refract the ray
        if (randomNum > fresnelFactor) {
            // Calculate refraction ratio based on material's index of refraction
            float ratio = 1.0f / m.indexOfRefraction;

            // If the ray is exiting the surface, adjust the normal and refraction ratio
            if (cosTheta >= 0.0f) {
                newRayNormal = -newRayNormal;
                ratio = m.indexOfRefraction;
            }

            // Compute the refracted ray direction
            pathSegment.ray.direction = glm::refract(newRayDirection, newRayNormal, ratio);

            // Adjust the ray's origin to avoid precision issues
            pathSegment.ray.origin += pathSegment.ray.direction * 0.01f;
        }
        else {
            // Reflect the ray direction if Fresnel factor is large
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, newRayNormal);
        }

        // Accumulate the color based on the material's color
        pathSegment.color *= m.color;

    }
    else
    {
    //===================================================================================
    // DIFFUSE MATERIALS
    //===================================================================================
    // Calculate a new random direction within the hemisphere for the ray
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color;
    }
    
    //===================================================================================
    // ALL MATERIALS
    //===================================================================================
    // Decrease bounces remaining
    pathSegment.remainingBounces -= 1;
}
