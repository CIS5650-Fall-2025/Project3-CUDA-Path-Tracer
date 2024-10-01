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

__host__ __device__ void scatter_ray(
    PathSegment &path_segment,
    const ShadeableIntersection &intersection,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    const auto intersection_point = path_segment.ray.origin + path_segment.ray.direction * (intersection.t - EPSILON);

    thrust::uniform_real_distribution<float> u01{0, 1};
    // const auto probability = u01(rng);
    const auto probability = 1.0f;

    if (probability < m.hasReflective) {
        path_segment.ray.direction = glm::normalize(glm::reflect(path_segment.ray.direction, intersection.surfaceNormal));
        path_segment.ray.origin = intersection_point;
        path_segment.color *= m.specular.color;
    } else {
        path_segment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng));
        path_segment.ray.origin = intersection_point;
    }

    path_segment.color *= m.color;
}
