#include "interactions.h"
#include "materials.h"
#include "samplers.h"

// Convert input color from RGB to CIE XYZ
// and return the luminance value (Y)
// https://www.cs.rit.edu/~ncs/color/t_convert.html#RGB%20to%20XYZ%20&%20XYZ%20to%20RGB
__host__ __device__ float luminance(const glm::vec3& color)
{
    return color[0] * 0.212671f + color[1] * 0.715160f + color[2] * 0.072169f;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec2 texCoord,
    const Material &m,
    glm::vec4* textures,
    thrust::default_random_engine &rng)
{
    
    bool scattered;
    glm::vec3 bsdf;
    float pdf;

    glm::vec3 attenuation;

    // hopefully we can template if we have time later on
    glm::vec3 dirIn = pathSegment.ray.direction;

    glm::vec3 mColor = m.color;
    if (m.texType == 2) {
        glm::vec4 sampledColor = sampleBilinear(m.imageTextureInfo, texCoord, textures);

        // Do nothing with alpha channel for now
        mColor = glm::vec3(sampledColor);
    }

    switch (m.type) {
    case LAMBERTIAN:
        scattered = scatterLambertian(pathSegment, intersect, normal, m, rng);
        bsdf = evalLambertian(dirIn, pathSegment.ray.direction, normal, m, mColor);
        pdf = pdfLambertian(dirIn, pathSegment.ray.direction, normal);
        attenuation = bsdf / pdf;
        break;
    case METAL:
        scattered = scatterMetal(pathSegment, intersect, normal, m, rng);
        bsdf = evalMetal(dirIn, pathSegment.ray.direction, normal, m, mColor);
        attenuation = bsdf;
        break;
    case DIELECTRIC:
        scattered = scatterDielectric(pathSegment, intersect, normal, m, rng);
        bsdf = evalDielectric(dirIn, pathSegment.ray.direction, normal, m, mColor);
        attenuation = bsdf;
        break;
    case EMISSIVE:
        scattered = scatterEmissive(pathSegment, intersect, normal, m, rng);
        bsdf = evalEmissive(dirIn, pathSegment.ray.direction, normal, m, mColor);
        pdf = pdfEmissive(dirIn, pathSegment.ray.direction, normal);
        attenuation = bsdf / pdf;
        break;
    default:
        break;
    }

    pathSegment.color *= attenuation;

    // Russian roulette
    thrust::uniform_real_distribution<float> u01(0, 1);
    float lum = luminance(pathSegment.color);
    if (lum < 1.0f)
    {
        float q = max(0.05f, 1.0f - lum);
        if (u01(rng) < q) 
            pathSegment.remainingBounces = -1;
        else
            pathSegment.color /= (1 - q);
    }

    if (pathSegment.remainingBounces < 0 && m.type != EMISSIVE) {
        // did not reach a light till max depth, terminate path as invalid
        pathSegment.color = glm::vec3(0.0f);
    }
}