#pragma once
#include "utilities.h"

__inline__ __device__ float pow5(float x) {
	return x * x * x * x * x;
}

__inline__ __device__ glm::mat3 LocalToWorld(const glm::vec3& N) {
	glm::vec3 T, B;
	if (glm::abs(N.x) > glm::abs(N.y)) {
		T = glm::vec3(-N.z, 0, N.x) / glm::sqrt(N.x * N.x + N.z * N.z);
	}
	else {
		T = glm::vec3(0, N.z, -N.y) / glm::sqrt(N.y * N.y + N.z * N.z);
	}
	B = glm::cross(N, T);

	if (glm::dot(glm::cross(T, B), N) < 0) {
		B = -B;
	}

	return glm::mat3(T, B, N);
}

__inline__ __device__ float AbsCosTheta(glm::vec3 w)
{
	return fabsf(w.z);
}

__inline__ __device__ float AbsDot(glm::vec3 a, glm::vec3 b)
{
	return fabsf(glm::dot(a, b));
}

__inline__ __device__ glm::vec3 SchlickFresnel(glm::vec3 F0, float cosTheta)
{
	return 1.0f + (glm::vec3(1.0f) - F0) * pow5(1.0f - cosTheta);
}

__inline__ __device__ // Cosine-weighted hemisphere sampling implementation
glm::vec3 cosineSampleHemisphere(const glm::vec2& rnd_param_uv) {
	float u1 = rnd_param_uv.x;
	float u2 = rnd_param_uv.y;

	float r = sqrt(u1);
	float theta = 2.0f * PI * u2;

	float x = r * cos(theta);
	float y = r * sin(theta);
	float z = sqrt(1.0f - u1);

	return glm::vec3(x, y, z);
}

// diffuse
__inline__ __device__ float pdf_baseDiffuse(glm::vec3 wo)
{
	return AbsCosTheta(wo) / PI;
}

__inline__ __device__ glm::vec3 Sample_baseDiffuse(const Material& m, glm::vec3 wo, glm::vec3 wh, glm::vec3 wi)
{
	glm::vec3 F0 = glm::vec3(0.5) + 2 * m.roughness * Square(glm::dot(wh, wi));
	return m.color / PI * SchlickFresnel(F0, AbsCosTheta(wi)) * SchlickFresnel(F0, AbsCosTheta(wo)) * dot(wh, wi);
}




















// integrate the BRDF
__inline__ __device__ glm::vec3 Sample_disneyBSDF(const Material& m, const Ray& ray, const glm::vec3& normal, const glm::vec2& xi, glm::vec3& wi)
{
	glm::mat3 ltw = LocalToWorld(normal);
	glm::mat3 wtl = glm::transpose(ltw);
	glm::vec3 n = glm::vec3(0, 0, 1);
	glm::vec3 wo = normalize(wtl * -ray.direction);

	glm::vec3 wi_diffuse = normalize(cosineSampleHemisphere(xi));
	glm::vec3 fdiffuse = Sample_baseDiffuse(m, wo, normalize(wi_diffuse + wo), wi_diffuse);
	float fdiffuse_pdf = pdf_baseDiffuse(wi_diffuse);
	fdiffuse = fdiffuse / fdiffuse_pdf * AbsDot(wi_diffuse, n);
	wi = normalize(ltw * wi_diffuse);

	return fdiffuse;
}