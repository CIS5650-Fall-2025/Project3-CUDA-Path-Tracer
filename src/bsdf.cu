#include "BSDF.h"

// Math helper
__device__ float cosThetaUnit(const glm::vec3& w) {
	return w.z;
}

__device__ float absCosThetaUnit(const glm::vec3& w) {
	return fabs(w.z);
}

// Build local coordinate system with intersection surface normal as local z-axis
__device__ void makeCoordTrans(glm::mat3& o2w, const glm::vec3& n) {
	// normal as local z
	glm::vec3 z = glm::normalize(n);
	glm::vec3 h = z;

	// h as not parallel to z
	if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z)) h.x = 1.0f;
	else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z)) h.y = 1.0f;
	else h.z = 1.0f;

	// construct x-axis and y-axis
	glm::vec3 y = glm::normalize(glm::cross(h, z));
	glm::vec3 x = glm::normalize(glm::cross(z, y));

	// fill in o2w
	o2w[0] = x;
	o2w[1] = y;
	o2w[2] = z;
}

// Get reflection wi with wo in local coordinate system
__device__ glm::vec3 reflect(const glm::vec3& wo) {
	return glm::vec3(-wo.x, -wo.y, wo.z);
}

__device__ float schlickFresnel(float cosThetaI, float etaI, float etaT) {
	float r0 = (etaI - etaT) / (etaI + etaT);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * powf(1.0f - cosThetaI, 5.0f);
}

// Get refract wi with wi in local coordinate system using Snell's Law, check whether it's total internal reflection
__device__ bool refract(const glm::vec3& wo, glm::vec3& wi, float eta) {
	float cosThetaI = wo.z;
	float sin2ThetaI = 1.0f - cosThetaI * cosThetaI;
	float sin2ThetaT = eta * eta * sin2ThetaI;
	if (sin2ThetaT >= 1.0f) {
		return false;
	}
	float cosThetaT = sqrt(1.0f - sin2ThetaT);
	wi = eta * -wo + (eta * cosThetaI - cosThetaT) * glm::vec3(0.0f, 0.0f, 1.0f);
	return true;
}

__device__ bool Refract(glm::vec3 wi, glm::vec3 n, float eta, glm::vec3& wt) 
{
	float cosThetaI = dot(n, glm::normalize(wi));
	float sin2ThetaI = fmaxf(0.0f, 1 - cosThetaI * cosThetaI);
	float sin2ThetaT = eta * eta * sin2ThetaI;

	if (!(1.0f - sin2ThetaT)) return false;
	float cosThetaT = sqrt(1 - sin2ThetaT);
	wt = eta * (wi - n * cosThetaI) - n * cosThetaT;
	return true;
}

// Sample a cosine-weighted random unit direction in a hemisphere in local coordinate system
__device__ glm::vec3 cosineWeightedHemisphereSampler3D(float* pdf, thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	float xi1 = u01(rng);
	float xi2 = u01(rng);

	float r = sqrt(xi1);
	float theta = 2.0f * PI * xi2;
	*pdf = sqrt(1 - xi1) * ONE_OVER_PI;
	return glm::vec3(r * cos(theta), r * sin(theta), sqrt(1 - xi1));
}

// Sample a halfway unit vector in a hemisphere
__device__ glm::vec3 phoneSampleHemisphere(float shininess, float* pdf, thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	float xi1 = u01(rng);
	float xi2 = u01(rng);

	float cosTheta = powf(xi1, 1.0f / (shininess + 1.0f));
	float sinTheta = sqrt(fmax(0.0f, 1.0f - cosTheta * cosTheta));
	float phi = 2.0f * PI * xi2;
	*pdf = (shininess + 2.0f) * ONE_OVER_TWO_PI * powf(glm::max(cosTheta, 0.0f), shininess);
	return glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// Universal BSDF constructor
BSDF::BSDF(enum MaterialType type,
		   glm::vec3 albeto, glm::vec3 specularColor, glm::vec3 transmittanceColor, glm::vec3 emissiveColor,
		   float kd, float ks, float shininess, float ior)
	: type(type), 
	  albeto(albeto), specularColor(specularColor), transmittanceColor(transmittanceColor), emissiveColor(emissiveColor),
	  kd(kd), ks(ks), shininess(shininess), ior(ior) {}

// BSDF evaluation based on material type using wi and wo
__device__ glm::vec3 BSDF::f(const glm::vec3& wo, const glm::vec3& wi) {
	switch (type) {
	case MIRROR:
		// Perfect reflection
		return glm::vec3(0.0f);
	
	case REFRACT:
		// Perfect transmission
		return glm::vec3(0.0f);

	case GLASS:
		// reflect-refract using Schlick's approximation
		return glm::vec3(0.0f);

	case GLOSSY:
		// Blinn-Phone
		glm::vec3 diffuse = kd * albeto * ONE_OVER_PI;
		glm::vec3 h = glm::normalize(wi + wo);
		float specAngle = glm::max(h.z, 0.0f);
		glm::vec3 specular = ks * specularColor * (shininess + 2.0f) * ONE_OVER_TWO_PI * powf(specAngle, shininess);
		return diffuse + specular;

	case EMISSION:
		// Light
		return glm::vec3(0.0f);

	default:
		// Default Diffuse
		return albeto * ONE_OVER_PI;
	}
}

// BSDF sampling for wi using wo in local coordinate system and evaulation based on material type using wi and wo
__device__ glm::vec3 BSDF::sampleF(const glm::vec3& wo, 
								   glm::vec3& wi, float* pdf, 
								   thrust::default_random_engine& rng) {
	// Diffuse by default
	switch (type) {
	case MIRROR:
		// Perfect reflection
		wi = reflect(wo);
		*pdf = 1;
		return specularColor / absCosThetaUnit(wi);
	
	case REFRACT:
		/*float cosThetaI = wo.z;
		float eta = cosThetaI > 0 ? 1.0f / ior : ior;
		float sin2ThetaT = eta * eta * fmax(0.0f, (1.0f - cosThetaI * cosThetaI));

		if (sin2ThetaT >= 1.0f) {
			*pdf = 0.0f;
			return glm::vec3(0.0f);
		}
		else 
		{
			float cosThetaT = sqrtf(1.0f - sin2ThetaT);
			wi = cosThetaI > 0 ? glm::vec3(-eta * wo.x, -eta * wo.y, -cosThetaT) : glm::vec3(-eta * wo.x, -eta * wo.y, cosThetaT);
			*pdf = 1.0f;
			float transmissionFactor = (1.0f - schlickFresnel(fabsf(cosThetaI), 1.0f, ior)) / (eta * eta);
			return  transmittanceColor * (transmissionFactor / fabsf(cosThetaT));
		}*/

		float etaA = 1.f;
		float etaB = ior;
		bool entering = wo.z < 0;
		float etaI = entering ? etaA : etaB;
		float etaT = entering ? etaB : etaA;

		glm::vec3 N = entering ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 0.0f, -1.0f);
		wi = glm::reflect(wo, N);

		Refract(wo, N, etaI / etaT, wi);
		*pdf = 1.0f;
		return transmittanceColor;


	case GLASS:
		// reflect-refract using Schlick's approximation
		/*if (!refract(wo, wi, ior)) {
			reflect(wo, wi);
			*pdf = 1.0f;
			return specularColor / absCosThetaUnit(wi);
		}
		else 
		{
			float r0 = powf((1 - ior), 2) / powf((1 + ior), 2);
			float cosT = abs(wo.z);
			float r = r0 + (1 - r0) * powf((1 - cosT), 5);
			thrust::uniform_real_distribution<float> u01(0, 1);
			float p = u01(rng);
			if (p < r) {
				reflect(wo, wi);
				*pdf = r;
				return r * specularColor / absCosThetaUnit(wi);
			}
			else 
			{
				refract(wo, wi, ior);
				*pdf = 1 - r;
				float eta;
				if (wo.z > 0) {
					eta = 1.0f / eta;
				}
				else {
					eta = ior;
				}
				return (1 - r) * transmittanceColor / absCosThetaUnit(wi) / powf(eta, 2);
			}
		}*/

		// Calculate cos(theta_i) using the fixed normal [0, 0, 1]
		//float cosThetaI = glm::dot(wo, glm::vec3(0.0f, 0.0f, 1.0f));

		//// Determine if we are entering or exiting the material
		//float etaI = 1.0f;
		//float etaT = ior;
		//bool isEntering = cosThetaI > 0.0f;

		//if (!isEntering) {
		//	// If exiting, reverse the IORs and the normal is flipped
		//	cosThetaI = -cosThetaI;
		//	etaI = ior;
		//	etaT = 1.0f;
		//}

		//// Compute Fresnel reflectance using Schlick's approximation
		//float reflectanceRatio = schlickFresnel(cosThetaI, etaI, etaT);

		//// Randomly choose between reflection and refraction
		//thrust::uniform_real_distribution<float> u011(0, 1);
		//float randVal = u011(rng);
		//if (randVal < reflectanceRatio) {
		//	// Sample reflection
		//	wi = glm::reflect(-wo, glm::vec3(0.0f, 0.0f, 1.0f));
		//	*pdf = reflectanceRatio;
		//	return specularColor;
		//}
		//else {
		//	// Sample refraction
		//	bool refracted = refract(wo, wi, etaI / etaT);
		//	if (!refracted) {
		//		// If total internal reflection, treat it as reflection
		//		wi = glm::reflect(-wo, glm::vec3(0.0f, 0.0f, 1.0f));
		//		*pdf = 1.0f;  // Total internal reflection always occurs
		//		return specularColor;
		//	}
		//	else {
		//		*pdf = 1.0f - reflectanceRatio;
		//		return transmittanceColor;
		//	}
		//}

	case GLOSSY:
		// Blinn-Phone
		thrust::uniform_real_distribution<float> u01(0, 1);
		float probSpecular = ks / (kd + ks);
		float dice = u01(rng);

		if (dice > probSpecular) {
			wi = cosineWeightedHemisphereSampler3D(pdf, rng);
			return f(wo, wi);
		}
		else 
		{
			glm::vec3 h = phoneSampleHemisphere(shininess, pdf, rng);
			wi = 2.0f * glm::dot(h, wo) * h - wo;
			return f(wo, wi);
		}

	case EMISSION:
		// Light
		wi = cosineWeightedHemisphereSampler3D(pdf, rng);
		return glm::vec3(0.0f);

	default:
		// Default Diffuse
		wi = cosineWeightedHemisphereSampler3D(pdf, rng);
		return albeto * ONE_OVER_PI;
	}
}

// Get material type
__device__ MaterialType BSDF::getType() const {
	return type;
}

// Get emission
__device__ glm::vec3 BSDF::getEmission() const {
	return emissiveColor;
}
