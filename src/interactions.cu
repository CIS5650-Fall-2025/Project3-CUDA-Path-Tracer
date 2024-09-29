#include "interactions.h"
#include <device_launch_parameters.h>

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
    glm::vec3& intersect,
    glm::vec3& normal,
    const Material& m,
    thrust::default_random_engine &rng)
{
    // uniform random float generator for probability sampling
    thrust::uniform_real_distribution<float> u01(0, 1);
    float prob = u01(rng);

    // specular reflection (mirror/metal)
    if (m.hasReflective > 0.f && prob < m.hasReflective) {

        // reflect the ray direction around the surface normal for specular reflection
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);

    } // specular refraction (glass)
    else if (m.hasRefractive > 0.f) {

        // is the ray entering or exiting the material?
        bool isEntering = glm::dot(pathSegment.ray.direction, normal) < 0.0f;
        glm::vec3 correctedNormal = isEntering ? normal : -normal;

        // relative index of refraction (eta)
        float eta = isEntering ? (1.0f / m.indexOfRefraction) : m.indexOfRefraction;

        // Snell's Law
        glm::vec3 refractedDir = glm::refract(glm::normalize(pathSegment.ray.direction), correctedNormal, eta);
        
        if (glm::length(refractedDir) <= 0) { // total internal reflection occurs
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, correctedNormal);
        }
        else {
            // Fresnel reflection probability via Schlick's approximation
            float cosTheta = glm::dot(-glm::normalize(pathSegment.ray.direction), correctedNormal);
            float R0 = (1.0f - m.hasRefractive) / (1.0f + m.hasRefractive);
            float fresnelReflectance = (R0 * R0) + (1.0f - (R0 * R0)) * powf(1.0f - fabs(cosTheta), 5.0f);

            if (prob < fresnelReflectance) pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            else pathSegment.ray.direction = refractedDir; 
        }

    } // diffuse scattering (Lambertian reflection)
    else {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    }

    // fix shadow acne: offset points that are very close to calculated intersection
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;

}