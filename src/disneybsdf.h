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
__inline__ __device__ float pdf_baseDiffuse(const glm::vec3& wi)
{
	return AbsCosTheta(wi) / PI;
}

__inline__ __device__ glm::vec3 Sample_baseDiffuse(const Material& m, const glm::vec3& wo, const glm::vec3& wh, const glm::vec3& wi)
{
	glm::vec3 F0 = glm::vec3(0.5) + 2 * m.roughness * Square(glm::dot(wh, wi));
	return m.color / PI * SchlickFresnel(F0, AbsCosTheta(wi)) * SchlickFresnel(F0, AbsCosTheta(wo)) * AbsCosTheta(wi);
}


// Lommel-Seeliger subsurface scattering approximation  
__inline__ __device__ float Base_subsurface(float roughness, const glm::vec3& wi, const glm::vec3& wh)
{
	return roughness * Square(AbsDot(wi, wh));
}

__inline__ __device__ glm::vec3 Sample_subsurface(const Material& m, const glm::vec3& wo, const glm::vec3& wh, const glm::vec3& wi)
{
	glm::vec3 F0(Base_subsurface(m.roughness, wi, wh));
	return 1.65f * m.color / PI 
		* ((SchlickFresnel(F0, AbsCosTheta(wi))) * (SchlickFresnel(F0, AbsCosTheta(wo))) * (1.f / (AbsCosTheta(wi) + AbsCosTheta(wo)) - 0.5f) + 0.5f) 
		* AbsCosTheta(wi);
}

// Metal


// calculate microfacet normal
__inline__ __device__ glm::vec3 sampleGGXVNDF(glm::vec3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
	// Section 3.2: transforming the view direction to the hemisphere configuration
	glm::vec3 Vh = normalize(glm::vec3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
	// Section 4.1: orthonormal basis (with special case if cross product is zero)
	float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	glm::vec3 T1 = lensq > 0 ? glm::vec3(-Vh.y, Vh.x, 0) * 1.0f/ sqrt(lensq) : glm::vec3(1, 0, 0);
	glm::vec3 T2 = cross(Vh, T1);
	// Section 4.2: parameterization of the projected area
	float r = sqrt(U1);
	float phi = 2.0 * PI * U2;
	float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	float s = 0.5 * (1.0 + Vh.z);
	t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
	// Section 4.3: reprojection onto hemisphere
	glm::vec3 Nh = t1 * T1 + t2 * T2 + sqrt(glm::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
	// Section 3.4: transforming the normal back to the ellipsoid configuration
	glm::vec3 Ne = normalize(glm::vec3(alpha_x * Nh.x, alpha_y * Nh.y, glm::max(0.0f, Nh.z)));
	return Ne;
}


__inline__ __device__ float GGX_Distribution(const glm::vec3& wh, float roughness, float anistropic, float& ax, float& ay)
{
	float aspect = sqrt(1.0f - 0.9 * anistropic * anistropic);
	ax = glm::max(1e-4f, Square(roughness) / aspect);
	ay = glm::max(1e-4f, Square(roughness) * aspect);
	return 1.0f / (PI * ax * ay * Square(Square(wh.x / ax) + Square(wh.y / ay) + Square(wh.z)));
}

__inline__ __device__ float GGX_Smith(float ax, float ay, const glm::vec3& wl)
{
	float a2 = (sqrt((1.0f + (Square(wl.x * ax) + Square(wl.y * ay)) / Square(wl.z)) + 1.0f) - 1.0f) / 2.0f;
	return 1.0f / (1.0f + a2);
}

__inline__ __device__ float GGX_Geometry(const glm::vec3& wo, const glm::vec3& wi, float roughness, float ax, float ay)
{
	return GGX_Smith(ax, ay, wo) * GGX_Smith(ax, ay, wi);
}

// microfacet reflection pdf
__inline__ __device__ float pdf_microfacet(float D, float G, const glm::vec3 & wi, const glm::vec3& h)
{
	return D * G / (4 * AbsDot(wi, h));
}

__inline__ __device__ glm::vec3 Sample_metal(const Material& m, const glm::vec3& wo, const glm::vec3& wh, const glm::vec3& wi, float& pdf)
{
	float ax, ay; ax = ay = 0.0f;
	glm::vec3 F = SchlickFresnel(m.color, AbsDot(wi, wh));
	float D = GGX_Distribution(wh, m.roughness, m.anisotropic, ax, ay);
	float G = GGX_Geometry(wo, wh, m.roughness, ax, ay);

	pdf = pdf_microfacet(D, G, wi, wh);
	return D * F * G / (4 * AbsCosTheta(wo));
}















// integrate the BRDF
__inline__ __device__ glm::vec3 Sample_disneyBSDF(const Material& m, const glm::vec3& woW, const glm::vec2& xi, glm::vec3& wi, const glm::mat3& ltw, const glm::mat3& wtl, float& pdf)
{
	glm::vec3 n = glm::vec3(0, 0, 1);
	glm::vec3 wo = normalize(wtl * woW);

	// diffuse
	glm::vec3 wi_d = normalize(cosineSampleHemisphere(xi));
	glm::vec3 fd = Sample_baseDiffuse(m, wo, normalize(wi_d + wo), wi_d);
	float fd_pdf = pdf_baseDiffuse(wi_d);
	wi = normalize(ltw * wi_d);

	// subsurface scattering
	glm::vec3 fss = Sample_subsurface(m, wo, normalize(wi + wo), wi);
	fd = Lerp(fd, fss, m.subsurface);

	pdf = fd_pdf;
	return fd;
}

__inline__ __device__ glm::vec3 Evaluate_disneyBSDF(const Material& m, const glm::vec3& wiW, const glm::vec3& woW, const glm::mat3& ltw, const glm::mat3& wtl, float& pdf)
{
	glm::vec3 n = glm::vec3(0, 0, 1);
	glm::vec3 wi = normalize(wtl * wiW);
	glm::vec3 wo = normalize(wtl * woW);

	pdf = pdf_baseDiffuse(wi);

	// diffuse
	glm::vec3 fd = Sample_baseDiffuse(m, wo, normalize(wi + wo), wi);

	// subsurface scattering
	glm::vec3 fss = Sample_subsurface(m, wo, normalize(wi + wo), wi);
	fd = Lerp(fd, fss, m.subsurface);

	return fd;
}

__inline__ __device__ glm::vec3 SetUpBSDF(const Material& m, const glm::vec3& wo, float rand)
{

	return glm::vec3(0.0f);
}