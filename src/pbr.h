#pragma once
#include "utilities.h"

// brdf
__inline__ __device__ bool SameHemisphere(glm::vec3 v1, glm::vec3 v2) {
	return glm::dot(v1, v2) > 0;
}

__device__ glm::vec3 Sample_wh(glm::vec3 wo, glm::vec2 xi, float roughness) {
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

__device__ glm::vec3 fresnelSchlick(float cosTheta, glm::vec3 F0) {
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

__device__ float distributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = PI * denom * denom;

    return num / denom;
}

__device__ float geometrySchlickGGX(float NdotV, float roughness) {
	float r = (roughness + 1.0f);
	float k = (r * r) / 8.0f;

	float num = NdotV;
	float denom = NdotV * (1.0f - k) + k;

	return num / denom;
}

__device__ float geometrySmith(glm::vec3 N, glm::vec3 V, glm::vec3 L, float roughness) {
	float NdotV = max(dot(N, V), 0.0f);
	float NdotL = max(dot(N, L), 0.0f);
	float ggx2 = geometrySchlickGGX(NdotV, roughness);
	float ggx1 = geometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

// btdf 
__inline__ __device__ bool Refract(const glm::vec3& wo, const glm::vec3& n, float eta, glm::vec3& wi) {
    float cosThetaI = dot(n, wo);
    float sin2ThetaI = glm::max(0.f, 1.0f - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;

    if (sin2ThetaT >= 1.0f) {
        return false;
    }

    float cosThetaT = sqrt(1.0f - sin2ThetaT);
    wi = eta * -wo + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

__inline__ __device__ glm::vec3 btdf(glm::vec3 albedo, glm::vec3 nor, glm::vec3 wo, float ior,
    glm::vec3& wiW)
{
    // Hard-coded to index of refraction of glass
    float etaA = 1.;
    float etaB = ior;
    float eta;

    if (dot(wo, nor) < 0) {
        eta = etaB / etaA;  
    }
    else {
        eta = etaA / etaB;  
        nor = -nor;  
    }

    glm::vec3 wi;
    if (!Refract(wo, nor, eta, wi)) {
        wiW = wi;  
        return glm::vec3(0.0); 
    }

    wiW = wi;

    return albedo / glm::abs(glm::dot(nor, wi));
}

__inline__ __device__ float Tan2Theta(const glm::vec3& wh) {
    float cosTheta = wh.z;
    return (1 - cosTheta * cosTheta) / (cosTheta * cosTheta);

}

__inline__ __device__ float Cos2Theta(const glm::vec3& wh) {
    return wh.z * wh.z;
}

__inline__ __device__ float Cos2Phi(const glm::vec3& wh) {
    if (wh.x == 0 && wh.y == 0) return 1.0f;  
    return (wh.x * wh.x) / (wh.x * wh.x + wh.y * wh.y);
}

__inline__ __device__ float Sin2Phi(const glm::vec3& wh) {
    if (wh.x == 0 && wh.y == 0) return 0.0f;  // 避免除零
    return (wh.y * wh.y) / (wh.x * wh.x + wh.y * wh.y);
}

__inline__ __device__ float AbsCosTheta(const glm::vec3& wh) {
	return glm::abs(wh.z);
}

__inline__ __device__ glm::mat3 LocalToWorld(const glm::vec3& N) {
    glm::vec3 tangent, bitangent;

    if (fabs(N.z) > 0.9999f) {
        tangent = glm::vec3(1.0f, 0.0f, 0.0f); 
    }
    else {
        tangent = glm::normalize(glm::vec3(-N.y, N.x, 0.0f));
    }

    bitangent = glm::normalize(glm::cross(N, tangent));

    return glm::mat3(tangent, bitangent, N);
}

__inline__ __device__ float TrowbridgeReitzD(glm::vec3 wh, float roughness) {
    float tan2Theta = Tan2Theta(wh);
    if (isinf(tan2Theta)) return 0.f;

    float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

    float e = (Cos2Phi(wh) / (roughness * roughness) + Sin2Phi(wh) / (roughness * roughness)) *
        tan2Theta;
    return 1 / (PI * roughness * roughness * cos4Theta * (1 + e) * (1 + e));
}

__inline__ __device__ float TrowbridgeReitzPdf(glm::vec3 wh, float roughness) {
    return TrowbridgeReitzD(wh, roughness) * AbsCosTheta(wh);
}

__inline__ __device__ glm::vec3 cookTorranceBRDF(const Material mat, glm::vec3 V, glm::vec3 N, glm::vec2 xi, float isRefract, glm::vec3& wi) {
    glm::vec3 H = glm::normalize(Sample_wh(V, xi, mat.roughness));
    glm::vec3 L = glm::normalize(glm::reflect(V, H));
    float NdotL = max(glm::dot(N, L), 0.f);
    float NdotV = max(glm::dot(N, V), 0.f);
    float NdotH = max(glm::dot(N, H), 0.f);

    glm::vec3 F = fresnelSchlick(NdotL, glm::vec3(mat.metallic));
    float D = distributionGGX(NdotH, mat.roughness);
    float G = geometrySmith(N, V, L, mat.roughness);

    glm::vec3 specular = D * G * F / (4.0f * NdotV * NdotL + 0.001f); // Prevent division by zero
    glm::vec3 diffuse = (1.0f - F) * mat.color / PI;

    glm::vec3 refraceWi;
    glm::vec3 refraction = btdf(mat.color, N, V, mat.ior, refraceWi);
    
	// decide wi
	float fresnelProb = (F.x + F.y + F.z) / 3.0f;
	wi = isRefract > fresnelProb ? refraceWi : L;

    // cook torrance pdf
	glm::mat3 ltw = LocalToWorld(N);
	glm::vec3 wh = glm::normalize(ltw * H);
	float pdf = TrowbridgeReitzPdf(wh, mat.roughness);

    return H;
    return specular + diffuse * (1 - mat.refractive) + refraction * mat.refractive;
}

