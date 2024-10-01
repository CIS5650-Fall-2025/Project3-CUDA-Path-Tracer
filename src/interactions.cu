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
    ShadeableIntersection shadeableIntersection,
    const Material &m,
    glm::vec4* textures,
    ImageTextureInfo bgTextureInfo,
    thrust::default_random_engine &rng,
    glm::vec3* dev_img)
{
    // If there was no intersection, sample environment map
    if (shadeableIntersection.t < 0.f) {
        glm::vec3 color = glm::vec3(sampleEnvironmentMap(bgTextureInfo, pathSegment.ray.direction, textures));
        dev_img[pathSegment.pixelIndex] += pathSegment.color * color;
        pathSegment.remainingBounces = -2;
        return;
    }
    
    bool scattered = false;
    glm::vec3 bsdf = glm::vec3(0.0f);
    float pdf = 1.f;

    glm::vec3 attenuation = glm::vec3(0.0f);

    glm::vec3 normal = glm::normalize(shadeableIntersection.surfaceNormal);
    glm::vec2 texCoord = shadeableIntersection.texCoord;

    glm::vec3 dirIn = glm::normalize(pathSegment.ray.direction);

    glm::vec3 mColor = m.color;
    if (m.texType == 1) {
        float sinVal = sin(m.checkerScale * intersect[0]) * sin(m.checkerScale * intersect[1]) * sin(m.checkerScale * intersect[2]);
        mColor = (sinVal < 0) ? glm::vec3(1.f, 1.f, 1.f) : glm::vec3(0.2f, 0.3f, 0.1f);
    } else if (m.texType == 2) {
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
        dev_img[pathSegment.pixelIndex] += pathSegment.color *= attenuation;
        return;
        break;
    default:
        // Invalid material
        scattered = false;
        break;
    }

    if (!scattered) {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = -1;
        return;
    }

    pathSegment.color *= attenuation;

    // Russian roulette
    thrust::uniform_real_distribution<float> u01(0, 1);
    float lum = luminance(pathSegment.color);
    if (lum < 1.0f)
    {
        float q = max(0.05f, 1.0f - lum);
        float randu01 = u01(rng);
        if (randu01 < q) {
            // Terminated paths make no contribution
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = -1;
        }
        else
            pathSegment.color /= (1.0f - q);
    }
}