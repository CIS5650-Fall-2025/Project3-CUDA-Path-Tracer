#include "bxdf.h"

template <typename T>
__host__ __device__ void cudaSwap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

// Evaluates the Fresnel reflectance equation between 2 dielectric media, assuming that light is
// unpolarized. cosThetaI is the angle of incidence measured from the normal; etaI is the refraction
// index of the medium that light is traveling through before reaching the interface with the
// other medium, whose refraction index etaT is. (A refraction index is a property of the medium:
// the ratio of the speed of light in a vacuum to the speed of light through the medium. Refraction
// indices for dielectrics are assumed to be real numbers.)
__host__ __device__ float DielectricFresnel(float cosThetaI, float etaI, float etaT) 
{
	cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

    // Make sure that etaI is really the index of the incident medium and etaT the index of the transmitted
    // medium. cosThetaI was measured between the surface's normal and the direction vector of incidence: if
    // it's negative, the 2 are in opposite hemispheres, and the ray is hitting the surface of the medium
    // from within that surface's medium: this medium's refraction index should really be etaI and not etaT.
	if (cosThetaI <= 0.f)
    {
        // Incidence vector is not inside the transmission medium, but the incident one.
        cudaSwap(etaI, etaT);
		cosThetaI = glm::abs(cosThetaI);
	}

    // Compute cosThetaT using Snell's law: etaI*sinThetaI = etaT*sinThetaT.
    // sinThetaI and cosThetaT are computed using a trigonometric identity: sin(theta)^2 + cos(theta)^2 = 1.
	float sinThetaI = glm::sqrt(glm::max(0.0f, 1 - cosThetaI*cosThetaI));
	float sinThetaT = etaI / etaT * sinThetaI;
	if (sinThetaT >= 1)
    {
        // Total internal reflection: light grazes the boundary of a medium with lower refraction index.
        // Snell's law doesn't have a solution, so refraction can't occur: light gets reflected back into
        // the incident medium.
		return 1.0f;
	}

	float cosThetaT = glm::sqrt(glm::max(0.0f, 1 - sinThetaT*sinThetaT));

    // Evaluate Fresnel reflectance equation for the parallel polarized component of light.
	float parallelR = ((etaT*cosThetaI) - (etaI*cosThetaT)) / ((etaT*cosThetaI) + (etaI*cosThetaT));

    // Evaluate Fresnel reflectance equation for the perpendicular polarized component of light.
	float perpendicularR = ((etaI*cosThetaI) - (etaT*cosThetaT)) / ((etaI*cosThetaI) + (etaT*cosThetaT));

    // The reflectance of unpolarized light is the average of the parallel and perpendicular polarized reflectances.
	return (parallelR*parallelR + perpendicularR*perpendicularR) / 2;
}

__host__ __device__ void SpecularBRDF(
    PathSegment& pathSegment,
    const Material& material,
    glm::vec3 intersect,
    glm::vec3 normal
) {
	pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
	pathSegment.ray.origin = intersect + (.1f) * pathSegment.ray.direction;
    pathSegment.color *= material.specular.color;
}

__host__ __device__ void SpecularBTDF(
    PathSegment& pathSegment,
    const Material& material,
    glm::vec3 intersect,
    glm::vec3 normal
) {
	glm::vec3 wo = pathSegment.ray.direction;

	bool outbound = glm::dot(wo, normal) > 0.f;
	glm::vec3 n = normal * (outbound ? -1.f : 1.f);
	float eta = outbound ? material.indexOfRefraction : (1.f / material.indexOfRefraction);

	glm::vec3 wi = glm::refract(wo, n, eta);

	pathSegment.ray.direction = wi;
	pathSegment.ray.origin = intersect + (.1f) * pathSegment.ray.direction;
    pathSegment.color *= material.specular.color;
}

__host__ __device__ void DielectricBxDF(
    PathSegment& pathSegment,
    const Material& material,
    glm::vec3 intersect,
    glm::vec3 normal,
    thrust::default_random_engine& rng
) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 wo = -pathSegment.ray.direction;
    float NdotWo = glm::dot(normal, wo);
    
    bool outbound = NdotWo < 0.f;
    float etaI = outbound ? material.indexOfRefraction : 1.f;
    float etaT = outbound ? 1.f : material.indexOfRefraction;

    if (u01(rng) < DielectricFresnel(NdotWo, etaI, etaT)/glm::abs(NdotWo))
    {
        SpecularBRDF(pathSegment, material, intersect, normal);
    }
    else
    {
        SpecularBTDF(pathSegment, material, intersect, normal);
    }
}
