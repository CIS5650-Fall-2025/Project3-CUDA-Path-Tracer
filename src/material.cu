#include "material.h"
#include "mathUtils.h"


__device__ glm::vec3 Material::lambertianSamplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf)
{
	glm::mat3 TBN = math::getTBN(nor);
	wi = math::sampleHemisphereCosine(rng.x, rng.y);
	*pdf = math::clampCos(wi) * INV_PI;
	// to world space
	wi = TBN * wi;
	return albedo * INV_PI;
}

__device__ glm::vec3 Material::samplef(const glm::vec3& nor, glm::vec3& wo, glm::vec3& wi, glm::vec3 rng, float* pdf)
{
	switch (type)
	{
	case Lambertian:
		return lambertianSamplef(nor, wo, wi, rng, pdf);
		break;
	default:
		return lambertianSamplef(nor, wo, wi, rng, pdf);
		break;
	}
}
