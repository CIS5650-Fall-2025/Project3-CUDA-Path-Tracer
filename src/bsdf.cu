#include "bsdf.h"

/** Helper Functions */
/** \brief Assuming that the given direction is in the local coordinate 
     * system, return the cosine of the angle between the normal and v */
__host__ __device__ float cosTheta(const glm::vec3 &v) {
    return v.z;
}

__host__ __device__ float tanTheta(const glm::vec3 &v) {
    float temp = 1 - v.z * v.z;

    if (temp <= 0.0f) {
        return 0.0f;
    }
        
    return sqrt(temp) / v.z;
}

__host__ __device__ void coordinateSystem(const glm::vec3 &v1, glm::vec3 &v2, glm::vec3 &v3) {
    if (abs(v1.x) > abs(v1.y)) {
        v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    }
    else {
        v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    }
            
    v3 = cross(v1, v2);
}

__host__ __device__ glm::mat3 LocalToWorld(const glm::vec3 &nor) {
    glm::vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

__host__ __device__ glm::mat3 WorldToLocal(const glm::vec3 &nor) {
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

__host__ __device__ float computeG(const glm::vec3 &v, const glm::vec3 &wh, const float roughness) {
    float v_dot_wh = glm::dot(v, wh); 
    float c = v_dot_wh / cosTheta(v);

    if (c <= 0) {
        return 0.0f;
    }

    float b = 1.0f / (roughness * tanTheta(v));
    if (b < 1.6) {
        float b_sqr = b * b;
        float nom = 3.535 * b + 2.181 * b_sqr;
        float denom = (1 + 2.276 * b + 2.577 * b_sqr);
        return nom / denom;
    }
        
    return 1.0f;
}
/*****************************************************************************/

/** PDFs */
__host__ __device__ float pdfDiffuse(const glm::vec3 &woL, const glm::vec3 &wiL){
    float cosThetaWiL = cosTheta(wiL);

    if (cosTheta(woL) <= 0 || cosThetaWiL <= 0) {
        return 0.0f;
    }

    return M_1_PIf * cosThetaWiL;
}

__host__ __device__ float pdfMirror(){
    return 0.0f;
}

__host__ __device__ float pdfDielectric(){
    return 0.0f;
}

__host__ __device__ float pdfMicrofacet(const float m_ks, const float roughness, const glm::vec3 &woL, const glm::vec3 &wiL){
    if (cosTheta(woL) <= 0 || cosTheta(woL) <= 0) {
        return .0f;
    }

    glm::vec3 wh = glm::normalize(woL + wiL);
    float D = squareToBeckmannPdf(wh, roughness);
    float Jh = 1.0f / (4 * glm::dot(wh, wiL));

    return m_ks * D * Jh + (1 - m_ks) * cosTheta(wiL) / M_PIf;
}
/*****************************************************************************/

/** Eval */
__host__ __device__ glm::vec3 evalMicrofacet(const glm::vec3 &woL, const glm::vec3 &wiL, const float roughness, const float m_extIOR, const float m_intIOR, const glm::vec3 &m_kd, const float m_ks) {
    if (cosTheta(woL) <= 0 || cosTheta(wiL) <= 0) {
        return glm::vec3(0.0f);
    }

    glm::vec3 wh = glm::normalize(woL + wiL);
    float wh_dot_woL = glm::dot(wh, woL);

    float D = squareToBeckmannPdf(wh, roughness);
    float F = fresnel(wh_dot_woL, m_extIOR, m_intIOR);
    float G = computeG(woL, wh, roughness) * computeG(wiL, wh, roughness);

    return m_kd / M_PIf + m_ks * D * F * G / (4 * cosTheta(woL) * cosTheta(wiL) * cosTheta(wh));
}

/** Bounce Directions and Return Colours */
__host__ __device__ glm::vec3 sampleDiffuse(const glm::vec3 &albedo, const glm::vec3 &normal, const glm::vec2 &sample2D, glm::vec3 &wiW) {
    wiW = squareToCosineHemisphere(sample2D, normal);

    return albedo;
}

__host__ __device__ glm::vec3 sampleMirror(const glm::vec3 &normal, const glm::mat3 &worldToLocal, const glm::vec3 &woW, glm::vec3 &wiW) {
    glm::vec3 woL = worldToLocal * woW;

    if (cosTheta(woL) <= 0.0f) {
        wiW = glm::vec3(0.0f);
    }

    wiW = glm::reflect(woW, normal);

    return glm::vec3(1.0f);
}

__host__ __device__ glm::vec3 sampleDielectric(const glm::vec3 &normal, const glm::mat3 &worldToLocal, const glm::mat3 &localToWorld, const glm::vec3 &woW, const float sample1D, const float m_extIOR, const float m_intIOR, glm::vec3 &wiW) {
    // Precompute WorldToLocal transformation and its inverse
    glm::vec3 woL = worldToLocal * woW;
    
    float Fr = fresnel(cosTheta(woL), m_extIOR, m_intIOR);
    float Ft = 1- Fr;

    // Check first whether we are returning a reflection direction
    bool reflection_more_likely = Fr > Ft;
    thrust::minimum<float> min_op;
    bool is_larger_than_smaller_prob = sample1D > min_op(Fr, Ft);
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

__host__ __device__ glm::vec3 sampleMicrofacet(const glm::vec3 &normal, const glm::mat3 &worldToLocal, const glm::mat3 &localToWorld, const glm::vec3 &woW, const glm::vec3 &m_kd, const float m_ks, const float roughness, const float m_extIOR, const float m_intIOR, const glm::vec2 sample2D, glm::vec3 &wiW) {
    glm::vec3 woL = worldToLocal * woW;
    glm::vec3 wiL;
    glm::vec2 sample;

    if (sample2D.x > m_ks) {
        sample = glm::vec2((sample2D.x - m_ks) / (1 - m_ks), sample2D.y);
        wiW = squareToCosineHemisphere(sample, normal);
        wiL = worldToLocal * wiW;
    }
    else {
        sample = glm::vec2(sample2D.x / m_ks, sample2D.y);
        glm::vec3 n = squareToBeckmann(sample, roughness);
        wiL = glm::reflect(woL, n);
        wiW = localToWorld * wiL;
    }

    if (cosTheta(wiL) <= 0.0f || cosTheta(woL) <= 0.0f) {
        return glm::vec3(0.0f);
    }

    return evalMicrofacet(woL, wiL, roughness, m_extIOR, m_intIOR, m_kd, m_ks) * cosTheta(wiL) / pdfMicrofacet(m_ks, roughness, woL, wiL);
}
/*****************************************************************************/