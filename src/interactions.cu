#include "interactions.h"
#include "materials.h"
#include "samplers.h"

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    
    bool scattered;
    glm::vec3 bsdf;
    float pdf;

    glm::vec3 attenuation;

    // hopefully we can template if we have time later on
    glm::vec3 dirIn = pathSegment.ray.direction;

    switch (m.type) {
    case LAMBERTIAN:
        scattered = scatterLambertian(pathSegment, intersect, normal, m, rng);
        bsdf = evalLambertian(dirIn, pathSegment.ray.direction, normal, m);
        pdf = pdfLambertian(dirIn, pathSegment.ray.direction, normal);
        attenuation = bsdf / pdf;
        break;
    case METAL:
        scattered = scatterMetal(pathSegment, intersect, normal, m, rng);
        bsdf = evalMetal(dirIn, pathSegment.ray.direction, normal, m);
        attenuation = bsdf;
        break;
    case DIELECTRIC:
        scattered = scatterDielectric(pathSegment, intersect, normal, m, rng);
        bsdf = evalDielectric(dirIn, pathSegment.ray.direction, normal, m);
        attenuation = bsdf;
        break;
    case EMISSIVE:
        scattered = scatterEmissive(pathSegment, intersect, normal, m, rng);
        bsdf = evalEmissive(dirIn, pathSegment.ray.direction, normal, m);
        pdf = pdfEmissive(dirIn, pathSegment.ray.direction, normal);
        attenuation = bsdf / pdf;
        break;
    default:
        break;
    }

    pathSegment.color *= attenuation;

    if (pathSegment.remainingBounces < 0 && m.type != EMISSIVE) {
        // did not reach a light till max depth, terminate path as invalid
        pathSegment.color = glm::vec3(0.0f);
    }
}