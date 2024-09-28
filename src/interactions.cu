#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng));      // cos(theta)
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

    return up * normal + cos(around) * over * perpendicularDirection1 + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ Sample sampleLight(
    glm::vec3 viewPoint,
    const Geom &geom,
    const Material *materials,
    thrust::default_random_engine &rng)
{
    Material material = materials[geom.materialid];
    if (geom.type == SQUARE)
    {
        thrust::uniform_real_distribution<float> uSquareSide(-0.5, 0.5);
        glm::vec2 squarePoint = glm::vec2(uSquareSide(rng), uSquareSide(rng));
        glm::vec3 lightPoint = multiplyMV(geom.transform, glm::vec4(squarePoint, 0, 1));
        glm::vec3 r = lightPoint - viewPoint;
        glm::vec3 incomingDirection = glm::normalize(r);
        float pdfdA = 1.f / (geom.scale.x * geom.scale.y);
        float rSquare = dot(r, r);
        glm::vec3 normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(0, 0, 1, 0)));
        float pdfdw = rSquare / dot(-incomingDirection, normal) * pdfdA;

        return Sample{
            .incomingDirection = incomingDirection,
            .value = material.color * material.emittance,
            .pdf = pdfdw,
            .delta = false,
        };
    }
    else if (geom.type == CUBE)
    {
        thrust::uniform_int_distribution<int> u02(0, 2);
        int faceAxis = u02(rng);
        thrust::uniform_int_distribution<int> usign(-1, 1);
        int faceSign = usign(rng);

        glm::vec3 normalObj = glm::vec3();
        normalObj[faceAxis] = faceSign;

        thrust::uniform_real_distribution<float> uSquareSide(-0.5, 0.5);
        glm::vec3 lightPointObj = glm::vec3(uSquareSide(rng), uSquareSide(rng), uSquareSide(rng));
        lightPointObj[faceAxis] = 0.5f * faceSign;

        float area = geom.scale.x * geom.scale.y * geom.scale.z;
        area /= geom.scale[faceAxis];

        glm::vec3 lightPoint = multiplyMV(geom.transform, glm::vec4(lightPointObj, 1));
        glm::vec3 r = lightPoint - viewPoint;
        glm::vec3 incomingDirection = glm::normalize(r);

        float rSquare = dot(r, r);
        glm::vec3 normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(normalObj, 0)));

        // TODO: check: sign(cosTheta) to check self-occlusion?
        float cosTheta = dot(-incomingDirection, normalObj);
        float pdfdw = rSquare / (std::abs(cosTheta) * area * 6);

        return Sample {
            .incomingDirection = incomingDirection,
            .value = material.color * material.emittance,
            .pdf = std::abs(pdfdw),
            .delta = false};
    }
    else if (geom.type == SPHERE)
    {
        // Assumption: sampling happens from outside the sphere (will be the case for most geom)
        thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec3 originObj = multiplyMV(geom.inverseTransform, glm::vec4(viewPoint, 1));
        glm::vec3 lightPointObj = calculateRandomDirectionInHemisphere(originObj, rng);
        glm::vec3 lightPoint = multiplyMV(geom.transform, glm::vec4(lightPointObj, 1));
        glm::vec3 normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(lightPointObj, 0)));
        glm::vec3 r = lightPoint - viewPoint;
        glm::vec3 incomingDirection = glm::normalize(r);

        // TODO: Check math here for uneven scaling?
        float pdfdA = 2 * PI * geom.scale.x * geom.scale.y * geom.scale.z / glm::length(lightPointObj / geom.scale);
        float rSquare = dot(r, r);
        float pdfdw = rSquare / dot(-incomingDirection, normal) * pdfdA;

        return Sample{
            .incomingDirection = incomingDirection,
            .value = material.color * material.emittance,
            .pdf = pdfdw,
            .delta = false};
    }

    return Sample();
}

// Shamelessly copied from 461 code
__host__ __device__ float getFresnel(const Material& m, float cosThetaI) {
    float etaI = 1.f;
    float etaT = 1.55;
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    // entering the material
    if (!cosThetaI > 0.f) {
        std::swap(etaI, etaT);
        cosThetaI = -cosThetaI;
    }

    float sinThetaI = sqrt(std::max(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    float cosThetaT = sqrt(max(0.f, 1.f - sinThetaT * sinThetaT));

    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
               ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
               ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}

__host__ __device__ Sample sampleReflective(glm::vec3 specColor, glm::vec3 outgoingDirection, glm::vec3 normal) {
    return Sample{
            .incomingDirection = glm::reflect(outgoingDirection, normal),
            .value = specColor,
            .pdf = 1.f,
            .delta = true};
}

__host__ __device__ Sample sampleTransmissive() {}


__host__ __device__ Sample sampleBsdf(
    const Material &material,
    glm::vec3 normal,
    glm::vec3 outgoingDirection,
    thrust::default_random_engine &rng)
{
    if (material.hasReflective)
    {
        return sampleReflective(material.specular.color, outgoingDirection, normal);
    } 
    // else if (true) {
    //     float fresnel = getFresnel(material, glm::dot(normal, outgoingDirection));
    //     thrust::uniform_real_distribution<float> u01(0, 1);
    //     if (u01(rng) < 0.f) {

    //     }
    // }
    return Sample{
        .incomingDirection = calculateRandomDirectionInHemisphere(normal, rng),
        .value = material.color / PI,
        .pdf = 1 / PI,
        .delta = false};
}

__host__ __device__ glm::vec3 getBsdf(const Material &material, glm::vec3 normal, glm::vec3 incomingDirection, glm::vec3 outgoingDirection)
{
    if (material.hasReflective)
    {
        return glm::vec3(0);
    }

    return material.color / PI;
}

__host__ __device__ void scatterRay(
    PathSegment &pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    Sample sampleBsdfImportance = sampleBsdf(m, normal, pathSegment.ray.direction, rng);

    const float clipping_offset = 0.01f;
    pathSegment.ray.direction = sampleBsdfImportance.incomingDirection;
    pathSegment.ray.origin = intersect + sampleBsdfImportance.incomingDirection * clipping_offset;
    pathSegment.throughput *= sampleBsdfImportance.value / sampleBsdfImportance.pdf;
}
