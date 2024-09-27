#include "interactions.h"

/** Helper Functions */
/** \brief Assuming that the given direction is in the local coordinate 
     * system, return the cosine of the angle between the normal and v */
__host__ __device__ float cosTheta(const glm::vec3 &v) {
    return v.z;
}

__host__ __device__ void coordinateSystem(const glm::vec3 v1, glm::vec3 &v2, glm::vec3 &v3) {
    if (abs(v1.x) > abs(v1.y)) {
        v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    }
    else {
        v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    }
            
    v3 = cross(v1, v2);
}

__host__ __device__ glm::mat3 LocalToWorld(glm::vec3 nor) {
    glm::vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

__host__ __device__ glm::mat3 WorldToLocal(glm::vec3 nor) {
    return transpose(LocalToWorld(nor));
}

__host__ __device__ float fresnel(float cosThetaI, float extIOR, float intIOR) {
    float etaI = extIOR, etaT = intIOR;

    if (extIOR == intIOR)
        return 0.0f;

    /* Swap the indices of refraction if the interaction starts
       at the inside of the object */
    if (cosThetaI < 0.0f) {
        thrust::swap(etaI, etaT);
        cosThetaI = -cosThetaI;
    }

    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    float eta = etaI / etaT,
          sinThetaTSqr = eta*eta * (1-cosThetaI*cosThetaI);

    if (sinThetaTSqr > 1.0f)
        return 1.0f;  /* Total internal reflection! */

    float cosThetaT = sqrt(1.0f - sinThetaTSqr);

    float Rs = (etaI * cosThetaI - etaT * cosThetaT)
             / (etaI * cosThetaI + etaT * cosThetaT);
    float Rp = (etaT * cosThetaI - etaI * cosThetaT)
             / (etaT * cosThetaI + etaI * cosThetaT);

    return (Rs * Rs + Rp * Rp) / 2.0f;
}
/*****************************************************************************/

/** Bounce Directions and Return Colours */
__host__ __device__ glm::vec3 sampleDiffuse(const glm::vec3 albedo, const glm::vec3 normal, thrust::default_random_engine &rng, glm::vec3 &wiW) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 sample(u01(rng), u01(rng));

    wiW = squareToCosineHemisphere(sample, normal);

    return albedo;
}

__host__ __device__ glm::vec3 sampleMirror(const glm::vec3 normal, const glm::vec3 woW, glm::vec3 &wiW) {
    glm::vec3 woL = WorldToLocal(normal) * woW;
    if (cosTheta(woL) <= 0.0f) {
        wiW = glm::vec3(0.0f);
    }

    wiW = glm::reflect(woW, normal);

    return glm::vec3(1.0f);
}

__host__ __device__ glm::vec3 sampleDielectric(const glm::vec3 normal, const glm::vec3 woW, thrust::default_random_engine &rng, float m_extIOR, float m_intIOR, glm::vec3 &wiW) {
    // Precompute WorldToLocal transformation and its inverse
    glm::mat3 localToWorld = LocalToWorld(normal);
    glm::mat3 worldToLocal = WorldToLocal(normal);
    glm::vec3 woL = worldToLocal * woW;
    
    float Fr = fresnel(cosTheta(woL), m_extIOR, m_intIOR);
    float Ft = 1- Fr;

    thrust::uniform_real_distribution<float> u01(0, 1);
    float randVal = u01(rng);

    // Check first whether we are returning a reflection direction
    bool reflection_more_likely = Fr > Ft;
    thrust::minimum<float> min_op;
    bool is_larger_than_smaller_prob = randVal > min_op(Fr, Ft);
    bool reflect = is_larger_than_smaller_prob && reflection_more_likely || !is_larger_than_smaller_prob && !reflection_more_likely;
    if (reflect) {
        wiW = localToWorld * glm::vec3(-woL.x, -woL.y, woL.z);
        // In this case the return value is the same as that of mirror.cpp. Mathematically, it's F/F
        return glm::vec3(1.0f);
    }

    glm::vec3 n(0.0f, 0.0f, 1.0f);
    bool entering = cosTheta(woL) > 0;
    float eta1_eta2 = entering ? m_extIOR / m_intIOR : m_intIOR / m_extIOR;

    if (!entering) {
        // when the ray is shooting from the glass to the air
       n = -n;
    }

    float eta1_eta2_sq = eta1_eta2 * eta1_eta2;
    float woL_dot_n = glm::dot(woL, n);
    float woL_dot_n_sq = woL_dot_n * woL_dot_n;
    float sqrtTerm = sqrt(1.0f - eta1_eta2_sq * (1.0f - woL_dot_n_sq));

    wiW = localToWorld * (-eta1_eta2 * (woL - woL_dot_n * n) - n * sqrtTerm);

    // Mathematically, this is (1-F) * (etaO/etaI)^2 / (1 - F) = (etaO/etaI)^2
    return glm::vec3(1.0f / eta1_eta2_sq);
}

__host__ __device__ glm::vec3 sampleMicrofacet(const glm::vec3 normal, const glm::vec3 woW, const float m_ks, const float roughness, thrust::default_random_engine &rng, glm::vec3 &wiW) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 randVals = glm::vec2(u01(rng), u01(rng));

    glm::vec2 sample;
    if (randVals.x > m_ks) {}
    else {
        sample = glm::vec2(randVals.x / m_ks, randVals.y);
        
    }

    return glm::vec3(0.0f);
}
/*****************************************************************************/

/** PDFs */
__host__ __device__ float pdfDiffuse(const glm::vec3 woL, const glm::vec3 wiL){
    if (cosTheta(woL) <= 0 || cosTheta(wiL) <= 0) {
        return 0.0f;
    }

    return INV_PI * cosTheta(wiL);
}

__host__ __device__ float pdfMirror(){
    return 0.0f;
}

__host__ __device__ float pdfDielectric(){
    return 0.0f;
}

__host__ __device__ float pdfMicrofacet(const float m_ks, const float roughness, const glm::vec3 woL, const glm::vec3 wiL){
    if (cosTheta(woL) <= 0 || cosTheta(woL) <= 0) {
        return .0f;
    }

    glm::vec3 wh = glm::normalize(woL + wiL);
    float D = squareToBeckmannPdf(wh, roughness);
    float Jh = 1.0f / (4 * glm::dot(wh, wiL));

    return m_ks * D * Jh + (1 - m_ks) * cosTheta(wiL) / M_PI;
}
/*****************************************************************************/

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 woW,
    glm::vec3 normal, // Here normal is in world space
    glm::vec3 &wiW,
    float &pdf,
    glm::vec3 &c,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // A basic implementation of pure-diffuse shading will just call the
    // squareToCosineHemisphere defined above.
    glm::vec3 woL = WorldToLocal(normal) * woW; 
    if (m.type == DIFFUSE) {
        c = sampleDiffuse(m.color, normal, rng, wiW);
        pdf = pdfDiffuse(woL, WorldToLocal(normal) * wiW);
    }
    else if (m.type == MIRROR) {
        c = sampleMirror(normal, woW, wiW);
        pdf = pdfMirror();
    }
    else if (m.type == DIELECTRIC) {
        // For now we hardcode the IORs. This can be changed later.
        c = sampleDielectric(normal, woW, rng, 1.000277f, 1.5046f, wiW);
        pdf = pdfDielectric();
    }
    else if (m.type == MICROFACET) {

    }
}
