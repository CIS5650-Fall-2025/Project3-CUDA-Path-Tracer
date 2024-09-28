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
        // Picks randomly from 3 closest sides of the cube
        glm::vec3 viewpointObj = multiplyMV(geom.inverseTransform, glm::vec4(viewPoint, 1));
        thrust::uniform_real_distribution<float> uSquareSide(-0.5, 0.5);

        thrust::uniform_int_distribution<int> u02(0, 2);
        int axis = u02(rng);

        glm::vec3 normalObj = glm::vec3();
        glm::vec3 lightPointObj;
        float pdfdA = 1.f / (geom.scale.x * geom.scale.y * geom.scale.z);
        if (axis == 0) {
            normalObj.x = glm::sign(viewpointObj.x);
            lightPointObj = 0.5f * normalObj + glm::vec3(0, uSquareSide(rng), uSquareSide(rng));
            pdfdA *= geom.scale.x;
        } else if (axis == 1) {
            normalObj.y = glm::sign(viewpointObj.y);
            lightPointObj = 0.5f * normalObj + glm::vec3(uSquareSide(rng), 0, uSquareSide(rng));
            pdfdA *= geom.scale.y;
        } else {
            normalObj.z = glm::sign(viewpointObj.z);
            lightPointObj = 0.5f * normalObj + glm::vec3(uSquareSide(rng), uSquareSide(rng), 0);
            pdfdA *= geom.scale.z;
        }

        glm::vec3 lightPoint = multiplyMV(geom.transform, glm::vec4(lightPointObj, 1));
        glm::vec3 r = lightPoint - viewPoint;
        glm::vec3 incomingDirection = glm::normalize(r);

        float rSquare = dot(r, r);
        glm::vec3 normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(normalObj, 0)));

        float cosTheta = dot(-incomingDirection, normalObj);
        float pdfdw = rSquare / std::abs(cosTheta) * pdfdA;

        return Sample {
            .incomingDirection = incomingDirection,
            .value = (cosTheta > 0) ? (material.color * material.emittance) : glm::vec3(0, 1, 1),
            .pdf = std::abs(pdfdw) / 3,
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

__host__ __device__ Sample sampleBsdf(
    const Material &material,
    glm::vec3 normal,
    glm::vec3 outgoingDirection,
    thrust::default_random_engine &rng)
{
    if (material.hasReflective)
    {
        return Sample{
            .incomingDirection = glm::reflect(outgoingDirection, normal),
            .value = material.specular.color,
            .pdf = 1.f,
            .delta = true};
    }
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
