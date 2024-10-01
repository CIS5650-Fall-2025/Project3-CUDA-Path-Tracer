#include "interactions.h"

__device__ void squareToDiskConcentric(const glm::vec2 xi, glm::vec3& wi)
{
    //Remap to [-1, 1], [-1, 1]
    glm::vec2 offset = 2.f * xi - glm::vec2(1, 1);
    if (offset.x == 0 && offset.y == 0)
    {
        //Handle base case
        wi = glm::vec3(0);
    }

    // Apply concentric mapping to point
    float theta, r;
    if (abs(offset.x) > abs(offset.y)) {
        r = offset.x;
        theta = PI_OVER_FOUR * (offset.y / offset.x);
    }
    else {
        r = offset.y;
        theta = PI_OVER_TWO - PI_OVER_FOUR * (offset.x / offset.y);
    }
    wi = r * glm::vec3(cos(theta), sin(theta), 0);
}

__device__ void squareToHemisphereCosine(const glm::vec2 xi, glm::vec3 &wi) {
    squareToDiskConcentric(xi, wi);
    //Extrapolate z using x, y coords of the point, uniformly sampled at the base of the hemisphere!
    float z = sqrt(glm::max(0.f, 1.f - wi.x * wi.x - wi.y * wi.y));
    wi.z = z;
}

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
__device__ void f_diffuse(
    glm::vec3& f,
    const Material& m)
{
    f = INV_PI * m.color;
}

__device__ void pdf_diffuse(
    float& pdf, const glm::vec3& wi)
{
    pdf = INV_PI * AbsCosTheta(wi);
}

__device__ void sample_f_diffuse(
    PathSegment& pathSegment,
    float& pdf,
    glm::vec3& f,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    //0. rng gen
    thrust::uniform_real_distribution<float> u01(0, 1);
    const glm::vec2 xi = glm::vec2(u01(rng), u01(rng));
    //1. Generate wi (local space)
    glm::vec3 wi = glm::vec3(0);
    squareToHemisphereCosine(xi, wi);
    //2. Find f
    f_diffuse(f, m);

    //3. Find pdf
    pdf_diffuse(pdf, wi);
    //4. update wi
    pathSegment.ray.direction = wi;
}
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    //Update ray in pathSegment
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
}

__device__ void sample_f(
    PathSegment& pathSegment,
    float& pdf,
    glm::vec3& f,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    sample_f_diffuse(pathSegment, pdf, f, normal, m, rng);
    pathSegment.ray.direction = LocalToWorld(normal) * pathSegment.ray.direction;
    //Update ray in pathSegment
}