#include "ggx.h"
#include "utilities.h"

// Adapted from CIS 5610 / Pbrt

__device__ float AbsDot(glm::vec3 a, glm::vec3 b) {
    return glm::abs(glm::dot(a, b));
}

__device__ float CosTheta(glm::vec3 w) { return w.z; }
__device__ float Cos2Theta(glm::vec3 w) { return w.z * w.z; }
__device__ float AbsCosTheta(glm::vec3 w) { return glm::abs(w.z); }
__device__ float Sin2Theta(glm::vec3 w) {
    return max(0.f, 1.f - Cos2Theta(w));
}
__device__ float SinTheta(glm::vec3 w) { return glm::sqrt(Sin2Theta(w)); }
__device__ float TanTheta(glm::vec3 w) { return SinTheta(w) / CosTheta(w); }

__device__ float Tan2Theta(glm::vec3 w) {
    return Sin2Theta(w) / Cos2Theta(w);
}

__device__ float CosPhi(glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
}
__device__ float SinPhi(glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
}
__device__ float Cos2Phi(glm::vec3 w) { return CosPhi(w) * CosPhi(w); }
__device__ float Sin2Phi(glm::vec3 w) { return SinPhi(w) * SinPhi(w); }

__device__ float TrowbridgeReitzD(glm::vec3 wh, float roughness) {
    float tan2Theta = Tan2Theta(wh);
    if (isinf(tan2Theta)) return 0.f;

    float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

    float e =
        (Cos2Phi(wh) / (roughness * roughness) + Sin2Phi(wh) / (roughness * roughness)) *
        tan2Theta;
    return 1 / (PI * roughness * roughness * cos4Theta * (1 + e) * (1 + e));
}

__device__ float Lambda(glm::vec3 w, float roughness)
{
    float absTanTheta = abs(TanTheta(w));
    if (isinf(absTanTheta)) return 0.;

    // Compute alpha for direction w
    float alpha =
        sqrt(Cos2Phi(w) * roughness * roughness + Sin2Phi(w) * roughness * roughness);
    float alpha2Tan2Theta = (roughness * absTanTheta) * (roughness * absTanTheta);
    return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
}

__device__ float TrowbridgeReitzG(glm::vec3 wo, glm::vec3 wi, float roughness)
{
    return 1 / (1 + Lambda(wo, roughness) + Lambda(wi, roughness));
}

__device__ float TrowbridgeReitzPdf(glm::vec3 wo, glm::vec3 wh, float roughness)
{
    return TrowbridgeReitzD(wh, roughness) * AbsCosTheta(wh);
}

__device__ bool SameHemisphere(glm::vec3 w, glm::vec3 wp)
{
    return w.z * wp.z > 0;
}

__device__ glm::vec3 Sample_wh(glm::vec3 wo, glm::vec2 xi, float roughness) {
	float cosTheta = 0;
    float phi = TWO_PI * xi[1];
    // We'll only handle isotropic microfacet materials
    float tanTheta2 = roughness * roughness * xi[0] / (1.0f - xi[0]);
    cosTheta = 1 / sqrt(1 + tanTheta2);

    float sinTheta =
        sqrt(max(0.f, 1.f - cosTheta * cosTheta));

    glm::vec3 wh = glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    if (!SameHemisphere(wo, wh)) wh = -wh;

    return wh;
}

__device__ glm::vec3 f_microfacet_refl(glm::vec3 albedo, glm::vec3 wo, glm::vec3 wi, float roughness)
{
    float cosThetaO = AbsCosTheta(wo);
    float cosThetaI = AbsCosTheta(wi);
    glm::vec3 wh = wi + wo;
    // Handle degenerate cases for microfacet reflection
    if (cosThetaI == 0 || cosThetaO == 0) return glm::vec3(0.f);
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return glm::vec3(0.f);
    wh = normalize(wh);
    // TODO: Handle different Fresnel coefficients
    glm::vec3 F = glm::vec3(1.);//fresnel->Evaluate(glm::dot(wi, wh));
    float D = TrowbridgeReitzD(wh, roughness);
    float G = TrowbridgeReitzG(wo, wi, roughness);
    return albedo * D * G * F /
        (4 * cosThetaI * cosThetaO);
}
