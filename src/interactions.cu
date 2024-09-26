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

// Lambertian
__host__ __device__ void scatterLambertian(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    pathSegment.ray.origin = intersect + normal * EPSILON;
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);

    pathSegment.remainingBounces--;
}

__host__ __device__ glm::vec3 evalLambertian(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m)
{
    return m.color * max(0.0f, glm::dot(pathSegment.ray.direction, normal) / PI);
}

__host__ __device__ float pdfLambertian(
    PathSegment& pathSegment,
    glm::vec3 normal)
{
    return max(0.0f, glm::dot(pathSegment.ray.direction, normal) / PI);
}

// Emissive
__host__ __device__ void scatterEmissive(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    pathSegment.remainingBounces = -1;
}

__host__ __device__ glm::vec3 evalEmissive(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m)
{
    return m.color * m.emittance;
}

__host__ __device__ float pdfEmissive(
    PathSegment& pathSegment,
    glm::vec3 normal)
{
    return 1.0f;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{

    glm::vec3 bsdf;
    float pdf;

    // hopefully we can template if we have time later on
    switch (m.type) {
    case LAMBERTIAN:
        scatterLambertian(pathSegment, intersect, normal, rng);
        bsdf = evalLambertian(pathSegment, normal, m);
        pdf = pdfLambertian(pathSegment, normal);
        break;
    case DIELECTRIC:
        break;
    case EMISSIVE:
        scatterEmissive(pathSegment, intersect, normal, rng);
        bsdf = evalEmissive(pathSegment, normal, m);
        pdf = pdfEmissive(pathSegment, normal);
        break;
    default:
        break;
    }

    pathSegment.color *= bsdf / pdf;

    if (pathSegment.remainingBounces < 0 && m.type != EMISSIVE) {
        // did not reach a light till max depth, terminate path as invalid
        pathSegment.color = glm::vec3(0.0f);
    }
}