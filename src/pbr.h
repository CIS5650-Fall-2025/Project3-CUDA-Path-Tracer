#pragma once
#include "utilities.h"

__device__ glm::vec3 fresnelSchlick(float cosTheta, glm::vec3 F0) {
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

__device__ float distributionGGX(glm::vec3 N, glm::vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = 3.14159265359f * denom * denom;

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

__device__ glm::vec3 cookTorranceBRDF(const Material& material, glm::vec3 V, glm::vec3 L, glm::vec3 N, glm::vec3 F0) {
    glm::vec3 H = normalize(V + L); // Half vector

    // Cook-Torrance BRDF components
    float D = distributionGGX(N, H, material.roughness);
    float G = geometrySmith(N, V, L, material.roughness);
    glm::vec3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0);

    // Cook-Torrance BRDF
    float NdotL = max(dot(N, L), 0.0f);
    float NdotV = max(dot(N, V), 0.0f);
    glm::vec3 numerator = D * G * F;
    float denominator = 4.0f * NdotV * NdotL + 0.001f; // Prevent division by zero

    return numerator / denominator;
}
