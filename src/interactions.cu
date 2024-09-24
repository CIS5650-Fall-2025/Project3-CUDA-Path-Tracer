#include "interactions.h"

#if 0
__host__ __device__ float FresnelDielectricEval(float cosThetaI, float etaI, float etaT) {
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    bool entering = cosThetaI > 0.f;
    if (!entering) {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        cosThetaI = abs(cosThetaI);
    }
    float sinThetaI = glm::sqrt(glm::max(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    float cosThetaT = glm::sqrt(glm::max(0.f, 1.f - sinThetaT * sinThetaT));

    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
}
#endif

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
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

// From 4610
// Faster Schlick Fresnel approximation than FresnelDielectricEval
__host__ __device__ float schlickFresnel(const glm::vec3& V, const glm::vec3& N, const float ior)
{
    float cosTheta = abs(glm::dot(V, N));
    float R0 = (1 - ior) / (1 + ior);
    R0 = R0 * R0;
    return R0 + (1 - R0) * pow(1.f - cosTheta, 5.0f);
}

//DIFFUSE_REFL
__host__ __device__ void Sample_f_diffuse(
    glm::vec3& wi,
    float& pdf,
    glm::vec3& bsdf,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    wi = calculateRandomDirectionInHemisphere(normal, rng);
    float cosTheta = glm::abs(glm::dot(wi, normal));
    pdf = cosTheta * INV_PI;
    if (pdf < EPSILON) {
        bsdf = glm::vec3(0.0f);
        return;
    }
    else {
        //bsdf = m.color * INV_PI;
        bsdf = m.color;
    }
}

//SPEC_REFL
__host__ __device__ void Sample_f_specular_refl(
    glm::vec3& wi,
    glm::vec3& wo,
    float& pdf,
    glm::vec3& bsdf,
    glm::vec3 normal,
    const Material& m,
    bool glass) {
    //wi = glm::reflect(wo, normal);
    float AbsCosTheta = glm::abs(glm::dot(wi, normal));
    pdf = 1.0f;
    bsdf = AbsCosTheta == 0 ? m.specular.color : m.specular.color / AbsCosTheta;
    float fresnel = schlickFresnel(wi, normal, m.indexOfRefraction);
    //bsdf = m.specular.color * fresnel;
    //bsdf = m.specular.color;
    if (glass)
    {
        bsdf = m.specular.color * fresnel;
    }
    else {
        bsdf = m.specular.color;
    }
    wi = glm::reflect(wo, normal);
}

//SPEC_TRAN
__host__ __device__ void Sample_f_specular_trans(
    glm::vec3& wi,
    glm::vec3& wo,
    float& pdf,
    glm::vec3& bsdf,
    glm::vec3 normal,
    const Material& m,
    bool glass) {
    float etaI = 1.0f, etaT = m.indexOfRefraction;
    float cosThetaI = glm::dot(wo, normal);

    if (cosThetaI < 0) {
        cosThetaI = -cosThetaI;
    }
    else {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        normal = -normal;
    }

    // Compute refraction
    wi = glm::refract(wo, normal, etaI / etaT);

    if (glm::length(wi) == 0) {
        wi = glm::reflect(wo, normal);  // Reflect instead
    }

    float fresnel = schlickFresnel(wo, normal, m.indexOfRefraction);

    if (glass) {
        bsdf = m.specular.color * (1.0f - fresnel);
    }
    else {
        bsdf = m.specular.color;
    }
    float AbsDot = glm::abs(glm::dot(wi, normal));
    pdf = 1.0f;
    bsdf = AbsDot == 0 ? bsdf : bsdf / AbsDot;
}

//SPEC_GLASS
__host__ __device__ void Sample_f_glass(
    glm::vec3& wi,
    glm::vec3& wo,
    float& pdf,
    glm::vec3& bsdf,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    float cosThetaI = glm::dot(wo, normal);
    float fresnel = schlickFresnel(wo, normal, m.indexOfRefraction);
    if (u01(rng) < fresnel) {
        //if (u01(rng) < 0.5f) {
        Sample_f_specular_refl(wi, wo, pdf, bsdf, normal, m, true);
    }
    else {
        Sample_f_specular_trans(wi, wo, pdf, bsdf, normal, m, true);
    }
    pdf = 1.0f;
    //bsdf *= 2.0f;
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 wi(0.f);
    glm::vec3 bsdf(0.f);
    float pdf = 1.0f;
    thrust::uniform_real_distribution<float> u01(0, 1);
    if (m.hasReflective > 0.0f && m.hasRefractive > 0.0f) {
        //write a glass material without using function
        float fresnel = schlickFresnel(pathSegment.ray.direction, normal, m.indexOfRefraction);

        // Fresnel-based reflection or refraction
        if (u01(rng) < fresnel) {
            // Reflect the ray if Fresnel reflection dominates
            Sample_f_specular_refl(wi, pathSegment.ray.direction, pdf, bsdf, normal, m, true);
        }
        else {
            // Refract the ray if Fresnel refraction dominates
            Sample_f_specular_trans(wi, pathSegment.ray.direction, pdf, bsdf, normal, m, true);
        }

        pathSegment.ray.origin = intersect;
    }
    else if (m.hasRefractive > 0.0f) {
        Sample_f_specular_trans(wi, pathSegment.ray.direction, pdf, bsdf, normal, m, false);
        pathSegment.ray.origin = intersect + EPSILON * normal;
    }
    else if (m.hasReflective > 0.0f) {
        // Specular reflection(mirror)
        Sample_f_specular_refl(wi, pathSegment.ray.direction, pdf, bsdf, normal, m, false);
        pathSegment.ray.origin = intersect + EPSILON * normal;
        //wi = glm::reflect(pathSegment.ray.direction, normal);
        //bsdf = m.specular.color;
    }
    else {
        // Diffuse reflection
        Sample_f_diffuse(wi, pdf, bsdf, intersect, normal, m, rng);
        pathSegment.ray.origin = intersect + EPSILON * normal;
    }

    // Update ray direction and origin
    pathSegment.ray.direction = glm::normalize(wi);
    //pathSegment.ray.origin = intersect + EPSILON * normal;

    pathSegment.color *= bsdf;
    pathSegment.remainingBounces--;
}
