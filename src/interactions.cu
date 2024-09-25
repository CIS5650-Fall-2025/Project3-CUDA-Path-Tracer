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

__host__ __device__ BsdfSample sampleLight(
    const Geom *geoms,
    size_t geomsSize,
    size_t index,
    glm::vec3 origin,
    size_t numLights,
    const Material& material,
    thrust::default_random_engine& rng
) {
    glm::vec3 lightPoint;
    BsdfSample result;
    const Geom& geom = geoms[index];
    if (geom.type == SQUARE) {
        thrust::uniform_real_distribution<float> uSquareSide(-0.5, 0.5);
        glm::vec2 squarePoint = glm::vec2(uSquareSide(rng), uSquareSide(rng));
        lightPoint = multiplyMV(geom.transform, glm::vec4(squarePoint, 0, 1));
        glm::vec3 r = lightPoint - origin;
        glm::vec3 wi = glm::normalize(r);
        float pdfdA = 1.f / (geom.scale.x * geom.scale.y);
        float rSquare = dot(r, r);
        glm::vec3 normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(0, 0, 1, 0)));
        float pdfdw = rSquare / dot(-wi, normal) * pdfdA;

        result = BsdfSample {
            .bsdf = material.color * material.emittance,
            .pdf = pdfdw / numLights,
            .delta = false,
            .wi = wi
        };

    } else if (geom.type == CUBE) {
        // Works by picking a side at random. Maybe it would be better to weight by area?
        glm::vec3 originObj = multiplyMV(geom.transform, glm::vec4(origin, 1));

        thrust::uniform_real_distribution<float> uSquareSide(-0.5, 0.5);
        glm::vec2 squarePoint = glm::vec2(uSquareSide(rng), uSquareSide(rng));
        thrust::uniform_int_distribution<int> uDimension(0, 2);
        int dimension = uDimension(rng);
        
        glm::vec3 lightPointObj;
        // TODO: remove slow modulo
        lightPointObj[dimension] = 0.5 * glm::sign(originObj[dimension]);
        lightPointObj[(1 + dimension) % 3] = squarePoint.x;
        lightPointObj[(2 + dimension) % 3] = squarePoint.y;

        lightPoint = multiplyMV(geom.transform, glm::vec4(lightPointObj, 1));
        glm::vec3 r = lightPoint - origin;
        glm::vec3 wi = glm::normalize(r);
        
        float sideArea = geom.scale.x * geom.scale.y * geom.scale.z / geom.scale[dimension];
        float pdfdA = 1.f / sideArea;

        float rSquare = dot(r, r);
        glm::vec3 normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(0, 0, 1, 0)));
        float pdfdw = rSquare / dot(-wi, normal) * pdfdA;

        // TODO: shadow checking
        result = BsdfSample {
            .bsdf = material.color * material.emittance,
            .pdf = pdfdw / (numLights * 3),
            .delta = false,
            .wi = wi
        };
        
    } else if (geom.type == SPHERE) {
        // Assumption: sampling happens from outside the sphere (will be the case for most geom)
        thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec3 originObj = multiplyMV(geom.transform, glm::vec4(origin, 1));
        glm::vec3 lightPointObj = calculateRandomDirectionInHemisphere(originObj, rng);
        lightPoint = multiplyMV(geom.inverseTransform, glm::vec4(lightPointObj, 1));
        glm::vec3 normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(lightPointObj, 0)));
        glm::vec3 r = lightPoint - origin;
        glm::vec3 wi = glm::normalize(r);
        
        //TODO: Check math here for uneven scaling?
        float pdfdA = 2 * PI * geom.scale.x * geom.scale.y * geom.scale.z / glm::length(lightPointObj / geom.scale);
        float rSquare = dot(r, r);
        float pdfdw = rSquare / dot(-wi, normal) * pdfdA;

        result =  BsdfSample {
            .bsdf = material.color * material.emittance,
            .pdf = pdfdw / numLights,
            .delta = false,
            .wi = wi
        };
    }

    return result;
}

__host__ __device__ BsdfSample sampleDiffuse(glm::vec3 albedo, glm::vec3 normal, thrust::default_random_engine &rng) {
    return BsdfSample {
        .bsdf = albedo / PI,
        .pdf = 1 / PI,
        .delta = false,
        .wi = calculateRandomDirectionInHemisphere(normal, rng)
    };
}

__host__ __device__ BsdfSample sampleSpecular(glm::vec3 albedo, glm::vec3 normal, glm::vec3 wo) {
    return BsdfSample {
        .bsdf = albedo,
        .pdf = 1.f,
        .delta = true,
        .wi = glm::reflect(wo, normal)
    };
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
    thrust::uniform_real_distribution<float> u01(0, 1);

    BsdfSample sampleBsdfImportance;
    if (m.hasReflective) {
        sampleBsdfImportance = sampleSpecular(m.specular.color, normal, pathSegment.ray.direction);
    } else {
        sampleBsdfImportance = sampleDiffuse(m.color, normal, rng);
    }
    
    const float clipping_offset = 0.01f;
    pathSegment.ray.direction = sampleBsdfImportance.wi;
    pathSegment.ray.origin = intersect + sampleBsdfImportance.wi * clipping_offset;
    pathSegment.color *= sampleBsdfImportance.bsdf / sampleBsdfImportance.pdf;
    if (--pathSegment.remainingBounces == 0) {
        pathSegment.color = glm::vec3(0);
    }
}
