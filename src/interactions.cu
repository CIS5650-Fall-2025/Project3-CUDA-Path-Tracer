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

__host__ __device__ void sampleDiffuse(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    float prob)
{
   pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
   pathSegment.color *= m.color / prob;
}

__host__ __device__ void sampleRefl(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m,
    float prob)
{
    pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.color *= m.specular.color / prob;
}

__host__ __device__ void sampleRefract(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m,
    float prob)
{
    float etaA = 1.f;
    float etaB = m.indexOfRefraction;
    bool entering = (glm::dot(pathSegment.ray.direction, normal) < 0);
    float etaI = entering ? etaA : etaB;
    float etaT = entering ? etaB : etaA;

    glm::vec3 N = entering ? normal : -normal;

    glm::vec3 refractedDir = glm::refract(pathSegment.ray.direction, N, etaI / etaT);

    if (glm::length(refractedDir) > EPSILON) {
        pathSegment.ray.direction = refractedDir;
    }
    else {
        //total internal reflection
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= 0.f;
        return;
    }
    pathSegment.color *= m.specular.color / prob;
}

__host__ __device__ float fresnel(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m) 
{
    glm::vec3 V = -pathSegment.ray.direction;
    glm::vec3 N = normal;
    float cosTheta = abs(glm::dot(V, N));
    float R0 = (1 - m.indexOfRefraction) / (1 + m.indexOfRefraction);
    R0 = R0 * R0;
    return R0 + (1 - R0) * pow(1.f - cosTheta, 5.f);
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{

    pathSegment.remainingBounces--;
    float probDiffuse = 0.f;

    float totalIntensity = glm::length(m.color) + glm::length(m.specular.color);

    if (totalIntensity > 0.f) {
        probDiffuse = glm::length(m.color) / totalIntensity;
    }

    thrust::uniform_real_distribution<float> u01(0, 1);
    float rand = u01(rng);

    if (rand < probDiffuse) {
        //diffuse shading
        sampleDiffuse(pathSegment, normal, m, rng, probDiffuse);
    }
    else {
        //glass-like material
        if (m.hasReflective > 0.f && m.hasRefractive > 0.f) {
            float f = fresnel(pathSegment, normal, m);
            if (rand < f) {
                sampleRefl(pathSegment, normal, m, 1.f);
            }
            else {
                sampleRefract(pathSegment, normal, m, 1.f);
            }
        }
        else if (m.hasReflective > 0.f) {
            //reflection
            //divide color by 1 - probDiffuse
            sampleRefl(pathSegment, normal, m, 1.f - probDiffuse);
        }
        else if (m.hasRefractive > 0.f) {
            //refraction
            //divide color by 1 - probDiffsue
            sampleRefract(pathSegment, normal, m, 1.f - probDiffuse);
        }
        else {
            //diffuse shading
            //divide color by 1 - probDiffuse
            //this probably shouldn't ever happen if our material is valid
            //because this means that m.specular.color != 0 but hasReflective & hasRefractive == 0
            if (probDiffuse != 0.f) {
                sampleDiffuse(pathSegment, normal, m, rng, 1.f - probDiffuse);
            }
        }
    }
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
}
