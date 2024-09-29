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
    float light_intensity = 1.0f;
    if (pathSegment.remainingBounces == 0)
        return;
    glm::vec3 incident_vector = glm::normalize(pathSegment.ray.direction);
    thrust::uniform_real_distribution<float> u01(0, 1);
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal,rng);
    pathSegment.ray.origin = intersect + EPSILON * pathSegment.ray.direction;
    pathSegment.color *= light_intensity*m.color*glm::max(glm::dot(pathSegment.ray.direction,normal),0.0f)/PI;
    pathSegment.remainingBounces--;
}
// __host__ __device__ void scatterRay(
//     PathSegment & pathSegment,
//     glm::vec3 intersect,
//     glm::vec3 normal,
//     const Material &m,
//     thrust::default_random_engine &rng)
// {
//     // TODO: implement this.
//     // A basic implementation of pure-diffuse shading will just call the
//     // calculateRandomDirectionInHemisphere defined above.
//     float light_intensity = 1.0f;
//     glm::vec3 brdf;
//     float pdf;
//     glm::vec3 incident_vector = glm::normalize(pathSegment.ray.direction);
//     thrust::uniform_real_distribution<float> u01(0, 1);
//     if (pathSegment.remainingBounces == 0)
//         return;
//     if (u01(rng) < m.hasReflective)
//     {
//         pathSegment.ray.direction = glm::reflect(incident_vector, normal);
//         brdf = m.specular.color/PI;
//         pdf = 1.0f;
//     }
//     else
//     {

//         glm::vec3 scattered_ray = calculateRandomDirectionInHemisphere(normal,rng);
//         pathSegment.ray.direction = glm::normalize(scattered_ray);
//         brdf = m.color/PI;
//         pdf = glm::max(glm::dot(pathSegment.ray.direction,normal),0.0f)/PI;
//     }
//     pathSegment.ray.origin = intersect + EPSILON * pathSegment.ray.direction;
//     //pathSegment.throughput *= light/_intensity*brdf*glm::max(glm::dot(pathSegment.ray.direction,normal),0.0f);
//     pathSegment.throughput *= light_intensity*brdf*glm::max(glm::dot(pathSegment.ray.direction,normal),0.0f)/pdf;
//     // pathSegment.throughput = glm::clamp(pathSegment.throughput, glm::vec3(0.0f), glm::vec3(1.0f));
//     pathSegment.remainingBounces--;
// }
