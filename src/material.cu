#include "material.h"
#include "mathUtils.h"

__device__ inline glm::vec3 fresnelSchlick(glm::vec3 f0, float HoV)
{
	return f0 + (1.0f - f0) * glm::pow(1.f - HoV, 5.f);
}


__device__ inline float fresnelDielectric(float cosThetaI, float etaI, float etaT)
{
	if (cosThetaI < 0.f)
	{
		float tmp = etaI;
		etaI = etaT;
		etaT = tmp;
		cosThetaI = -cosThetaI;
	}

	float sinThetaI = glm::sqrt(1.f - cosThetaI * cosThetaI);
	float sinThetaT = etaI / etaT * sinThetaI;
	if (sinThetaT >= 1) return 1;

	float cosThetaT = glm::sqrt(1.f - sinThetaT * sinThetaT);
	float rparll = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
	float rperpe = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
	return (rparll * rparll + rperpe * rperpe) * 0.5f;
}

__device__ inline float trowbridgeReitzD(const glm::vec3& wh, float roughness)
{
	float a2 = roughness * roughness;
	float tan2 = math::tan2Theta(wh);
	//if (isinf(tan2)) return 0.f;

	float cos4Theta = math::cos2Theta(wh) * math::cos2Theta(wh);

	float e = (math::cos2Phi(wh) / a2 + math::sin2Phi(wh) / a2) * tan2;
	return 1.f / (PI * roughness * roughness * cos4Theta * (1.f + e) * (1.f + e));
}

// ref: https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#TrowbridgeReitzDistribution::Sample_wm
__device__ glm::vec3 Material::microfacetSamplWh(const glm::vec3& wo, const glm::vec2& rng)
{
	glm::vec3 wh = glm::normalize(glm::vec3(roughness * wo.x, roughness * wo.y, wo.z));
	
	glm::vec3 T = (wh.z < 0.99999f) ? glm::normalize(glm::cross(glm::vec3(0, 0, 1), wh)) : glm::vec3(1, 0, 0);
	glm::vec3 B = glm::cross(wh, T);

	glm::vec2 p = math::sampleUniformDisk(rng.x, rng.y);
	float h = glm::sqrt(1.f - p.x * p.x);
	p.y = glm::mix(h, p.y, (1.f + wh.z) * 0.5f);

	float pz = glm::sqrt(glm::max(0.f, 1.f - glm::dot(p, p)));
	glm::vec3 nh = p.x * T + p.y * B + pz * wh;
	return glm::normalize(glm::vec3(roughness * nh.x, roughness * nh.y, glm::max(1e-6f, nh.z)));
}

__device__ inline float lambda(const glm::vec3& w, float roughness)
{
	float absTanTheta = glm::abs(math::tanTheta(w));
	if (isinf(absTanTheta)) return 0.f;
	float alpha = glm::sqrt(math::cos2Phi(w) * roughness * roughness + math::sin2Phi(w) * roughness * roughness);
	float alpha2Tan2Theta = (roughness * absTanTheta) * (roughness * absTanTheta);
	return (-1.f + sqrt(1.f + alpha2Tan2Theta)) * 0.5f;
}

__device__ inline float G1(const glm::vec3& w, float roughness) { return 1.f / (1.f + lambda(w, roughness)); }

__device__ inline float trowbridgeReitzPdf(const glm::vec3& wo, const glm::vec3& wh, float roughness)
{
	return G1(wo, roughness) * trowbridgeReitzD(wh, roughness) * math::absDot(wo, wh) / glm::max(1e-9f, math::absCosTheta(wo));
}

__device__ inline float trowbridgeReitzG(const glm::vec3& wo, const glm::vec3& wi, float roughness) {
	//return G1(wo, roughness) * G1(wi, roughness);
	return 1.f / (1.f + lambda(wo, roughness) + lambda(wi, roughness));
}

__device__ inline float Material::microfacetPDF(const glm::vec3& wo, const glm::vec3& wh)
{
	//glm::vec3 wm = wh.z < 0.f ? -wh : wh;

	return G1(wo, roughness) * trowbridgeReitzD(wh, roughness) / glm::max(1e-9f, 4.f * math::absCosTheta(wo));
	//return trowbridgeReitzPdf(wo, wh, roughness) /
	//	glm::max(1e-9f, (4.f * math::absDot(wo, wh)));
}

__device__ glm::vec3 Material::lambertianSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf)
{
	glm::mat3 TBN = math::getTBN(nor);
	wi = math::sampleHemisphereCosine(rng.x, rng.y);
	*pdf = math::clampCos(wi) * INV_PI;
	// to world space
	wi = TBN * wi;
	return albedo * INV_PI;
}

__device__ glm::vec3 Material::specularSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf)
{
	*pdf = 1.f;
	wi = glm::reflect(wo, nor);
	return glm::vec3(1.f);
}

__device__ glm::vec3 Material::microfacetSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf)
{
	glm::mat3 TBN = math::getTBN(nor);
	glm::mat3 invTBN = glm::transpose(TBN);

	glm::vec3 woL = invTBN * (-wo);
	glm::vec3 wh = microfacetSamplWh(woL, glm::vec2(rng.x, rng.y));
	//glm::vec3 wh = glm::vec3(0.f, 0.f, 1.f);
	wi = glm::reflect(-woL, wh);

	//glm::vec3 F = albedo * fresnelDielectric(math::absDot(woL, wh), 1.f, ior);
	glm::vec3 F = fresnelSchlick(albedo, math::absDot(woL, wh));
	glm::vec3 bsdf = F * (trowbridgeReitzD(wh, roughness) * trowbridgeReitzG(woL, wi, roughness) /
		glm::max(1e-9f, (4.f * math::absCosTheta(wi) * math::absCosTheta(woL))));
	*pdf = microfacetPDF(woL, wh);
	wi = TBN * wi;
	return bsdf;
}

__device__ glm::vec3 Material::metallicWorkflowSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf)
{

	return glm::vec3(0);
}

__device__ glm::vec3 Material::samplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf)
{
	switch (type)
	{
	case Lambertian:
		return lambertianSamplef(nor, wo, wi, rng, pdf);
		break;
	case Specular:
		return specularSamplef(nor, wo, wi, rng, pdf);
		break;
	case Microfacet:
		return microfacetSamplef(nor, wo, wi, rng, pdf);
		break;
	default:
		return lambertianSamplef(nor, wo, wi, rng, pdf);
		break;
	}
}
