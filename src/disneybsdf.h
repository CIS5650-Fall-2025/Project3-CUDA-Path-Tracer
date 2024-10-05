#pragma once
#include "utilities.h"

__inline__ __device__ float pow5(float x) {
	return x * x * x * x * x;
}



__inline__ __device__ glm::vec3 SchlickFresnel(const glm::vec3& F0, float cosTheta)
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
__inline__ __device__ float pdf_baseDiffuse(const glm::vec3& wo)
{
	return AbsCosTheta(wo) / PI;
}

__inline__ __device__ glm::vec3 Sample_baseDiffuse(const Material& m, const glm::vec3& wo, const glm::vec3& wh, const glm::vec3& wi)
{
	glm::vec3 F0 = glm::vec3(0.5) + 2 * m.roughness * Square(glm::dot(wh, wi));
	return m.color / PI * SchlickFresnel(F0, AbsCosTheta(wi)) * SchlickFresnel(F0, AbsCosTheta(wo)) * dot(wh, wi);
}


// Lommel-Seeliger subsurface scattering approximation  
__inline__ __device__ float Base_subsurface(float roughness, const glm::vec3& wi, const glm::vec3& wh)
{
	return roughness * Square(AbsDot(wi, wh));
}

__inline__ __device__ glm::vec3 Sample_subsurface(const Material& m, const glm::vec3& wo, const glm::vec3& wh, const glm::vec3& wi)
{
	glm::vec3 F0(Base_subsurface(m.roughness, wi, wh));
	return 1.25f * m.color / PI 
		* ((SchlickFresnel(F0, AbsCosTheta(wi))) * (SchlickFresnel(F0, AbsCosTheta(wo))) * (1.f / (AbsCosTheta(wi) + AbsCosTheta(wo)) - 0.5f) + 0.5f) 
		* AbsCosTheta(wi);
}


















// integrate the BRDF
__inline__ __device__ glm::vec3 Sample_disneyBSDF(const Material& m, const Ray& ray, const glm::vec3& normal, const glm::vec2& xi, glm::vec3& wi, const glm::mat3& ltw, const glm::mat3& wtl)
{
	glm::vec3 n = glm::vec3(0, 0, 1);
	glm::vec3 wo = normalize(wtl * -ray.direction);

	// diffuse
	glm::vec3 wi_d = normalize(cosineSampleHemisphere(xi));
	glm::vec3 fd = Sample_baseDiffuse(m, wo, normalize(wi_d + wo), wi_d);
	float fd_pdf = pdf_baseDiffuse(wi_d);
	wi = normalize(ltw * wi_d);

	// subsurface scattering
	glm::vec3 fss = Sample_subsurface(m, wo, normalize(wi + wo), wi);
	fd = Lerp(fd, fss, m.subsurface);
	fd = fd / fd_pdf * AbsDot(wi_d, n);

	return fd;
}