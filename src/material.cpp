#include "material.h"

#pragma region MaterialHelpFunction

/*From CIS 561*/
__host__ __device__ float AbsDot(glm::vec3 a, glm::vec3 b) {
    return glm::abs(dot(a, b));
}

__host__ __device__ float CosTheta(glm::vec3 w) { return w.z; }
__host__ __device__ float Cos2Theta(glm::vec3 w) { return w.z * w.z; }
__host__ __device__ float AbsCosTheta(glm::vec3 w) { return glm::abs(w.z); }
__host__ __device__ float Sin2Theta(glm::vec3 w) {
    return max(0.f, 1.f - Cos2Theta(w));
}
__host__ __device__ float SinTheta(glm::vec3 w) { return glm::sqrt(Sin2Theta(w)); }
__host__ __device__ float TanTheta(glm::vec3 w) { return SinTheta(w) / CosTheta(w); }

__host__ __device__  float Tan2Theta(glm::vec3 w) {
    return Sin2Theta(w) / Cos2Theta(w);
}

__host__ __device__  float CosPhi(glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
}
__host__ __device__  float SinPhi(glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
}
__host__ __device__  float Cos2Phi(glm::vec3 w) { return CosPhi(w) * CosPhi(w); }
__host__ __device__  float Sin2Phi(glm::vec3 w) { return SinPhi(w) * SinPhi(w); }

__host__ __device__ bool SameHemisphere(glm::vec3 w, glm::vec3 wp) {
    return w.z * wp.z > 0;
}

__host__ __device__ glm::vec3 Sample_wh(glm::vec3 wo, glm::vec2 xi, float roughness) {
    glm::vec3 wh;

    float cosTheta = 0;
    float phi = TWO_PI * xi[1];
    // We'll only handle isotropic microfacet materials
    float tanTheta2 = roughness * roughness * xi[0] / (1.0f - xi[0]);
    cosTheta = 1 / sqrt(1 + tanTheta2);

    float sinTheta =
        sqrt(max(0.f, 1.f - cosTheta * cosTheta));

    wh = glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    if (!SameHemisphere(wo, wh)) wh = -wh;

    return wh;
}

__host__ __device__ float schlickG(float cosTheta, float alpha) {
    float a = alpha * .5f;
    return cosTheta / (cosTheta * (1.f - a) + a);
}

__host__ __device__ float smithG(float cosWo, float cosWi, float alpha) {
    return schlickG(glm::abs(cosWo), alpha) * schlickG(glm::abs(cosWi), alpha);
}

__host__ __device__ float GTR2Distrib(float cosTheta, float alpha) {
    if (cosTheta < 1e-6f) {
        return 0.f;
    }
    float aa = alpha * alpha;
    float nom = aa;
    float denom = cosTheta * cosTheta * (aa - 1.f) + 1.f;
    denom = denom * denom * PI;
    return nom / denom;
}

__host__ __device__ float GTR2Pdf(glm::vec3 n, glm::vec3 m, glm::vec3 wo, float alpha) {
    return GTR2Distrib(glm::dot(n, m), alpha) * schlickG(glm::dot(n, wo), alpha) *
        glm::dot(m, wo) / glm::dot(n, wo);
}

#pragma endregion