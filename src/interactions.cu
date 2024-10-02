#include "interactions.h"

// For now we hardcode the IORs. This can be changed later.
#define EXT_IOR 1.000277f
#define INT_IOR 1.5046f

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
    glm::mat3 worldToLocal = WorldToLocal(normal);
    glm::mat3 localToWorld = LocalToWorld(normal);
    glm::vec3 woL = worldToLocal * woW; 
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 sample2D(u01(rng), u01(rng));

    if (m.type == MatType::DIFFUSE) {
        c = sampleDiffuse(m.color, normal, sample2D, wiW);
        glm::vec3 wiL = worldToLocal * wiW;
        pdf = pdfDiffuse(woL, wiL);
    }
    else if (m.type == MatType::MIRROR) {
        c = sampleMirror(normal, worldToLocal, woW, wiW);
        pdf = pdfMirror();
    }
    else if (m.type == MatType::DIELECTRIC) {
        c = sampleDielectric(worldToLocal, localToWorld, woW, sample2D.x, EXT_IOR, INT_IOR, wiW);
        pdf = pdfDielectric();
    }
    else if (m.type == MatType::MICROFACET) {
        glm::vec3 albedo = m.color;
        float m_ks = 1.0f - glm::max(albedo.x, glm::max(albedo.y, albedo.z));;
        c = sampleMicrofacet(normal, worldToLocal, localToWorld, woW, albedo, m_ks, m.roughness, EXT_IOR, INT_IOR, sample2D, wiW);
        glm::vec3 wiL = worldToLocal * wiW;
        pdf = pdfMicrofacet(m_ks, m.roughness, woL, wiL);
    }
}

__host__ __device__ void eval(const Material& m, const glm::vec3 normal, const glm::vec3 &woW, const glm::vec3 &wiW, glm::vec3 &brdf, float &pdf) {
    glm::mat3 worldToLocal = WorldToLocal(normal);
    glm::vec3 woL = worldToLocal * woW;
    glm::vec3 wiL = worldToLocal * wiW;

    if (m.type == MatType::DIFFUSE) {
        brdf = evalDiffuse(m.color, woL, wiL);
        pdf = pdfDiffuse(woL, wiL);
    }
    else if (m.type == MatType::MIRROR) {
        brdf = evalMirror();
        pdf = pdfMirror();
    }
    else if (m.type == MatType::DIELECTRIC) {
        brdf = evalDielectric();
        pdf = pdfDielectric();
    }
    else if (m.type == MatType::MICROFACET) {
        glm::vec3 albedo = m.color;
        float m_ks = 1.0f - glm::max(albedo.x, glm::max(albedo.y, albedo.z));
        brdf = evalMicrofacet(woL, wiL, m.roughness, EXT_IOR, INT_IOR, albedo, m_ks);
        pdf = pdfMicrofacet(m_ks, m.roughness, woL, wiL);
    }
}