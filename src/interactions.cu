#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

	// The random generated direction is cosine weighted by sqrt the random number
    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

	// the final direction is a combination of a linear combination of the two perpendicular directions and the normal
    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__device__ glm::vec3 getEnvironmentalRadiance(glm::vec3 direction, cudaTextureObject_t envMap) {
    float theta = acosf(direction.y);         // θ
    float phi = atan2f(direction.z, direction.x); // φ
    if (phi < 0) phi += 2.0f * PI;

    float u = phi / (2.0f * PI);            // [0, 1]
    float v = theta / PI;                   // [0, 1]
	if (envMap == NULL) return glm::vec3(0.0f); // return black if no envMap (for debugging purposes
	float4 texel = tex2D<float4>(envMap, u, v);
	return glm::vec3(texel.x, texel.y, texel.z);
}

__device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    float t,
    glm::vec3 normal, 
	glm::vec2 uv,
    const Material &m,
    thrust::default_random_engine &rng)
{

	glm::vec3 wi = glm::vec3(0.0f);
	glm::vec3 col = glm::vec3(0.0f);

    // TODO: implement PBR model
 //   glm::vec3 L = slerp(glm::reflect(pathSegment.ray.direction, normal), calculateRandomDirectionInHemisphere(normal, rng), m.roughness);
	//L = glm::normalize(L);
	//glm::vec3 H = glm::normalize(L + pathSegment.ray.direction);
	//float NdotL = glm::dot(normal, L);
	//float NdotV = glm::dot(normal, -pathSegment.ray.direction);
	//float NdotH = glm::dot(normal, H);
    //glm::vec3 F = fresnelSchlick(NdotL, glm::vec3(m.metallic));
    //glm::vec3 kd = (1.0f - F) * m.color / PI;

    if (m.reflective == 1.0f)
    {
		// perfect reflection
		wi = glm::reflect(pathSegment.ray.direction, normal);
		col = m.color;
	}
    else
    {
        // Ideal diffuse
		wi = calculateRandomDirectionInHemisphere(normal, rng);
        col = m.color;
    }

    pathSegment.remainingBounces--;

#ifdef DEBUG_NORMAL
    col = glm::vec3(1.f);
    pathSegment.color = DEBUG_NORMAL ? (normal + 1.0f) / 2.0f : normal;
	pathSegment.remainingBounces = 0;
#elif defined(DEBUG_WORLD_POS)
	col = glm::vec3(1.f);
    pathSegment.color = glm::clamp(intersect, glm::vec3(0), glm::vec3(1.0f));
	pathSegment.remainingBounces = 0;
#elif defined(DEBUG_UV)
	col = glm::vec3(1.f);
	pathSegment.color = glm::vec3(uv, 0);
	pathSegment.remainingBounces = 0;
#endif

	pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = glm::normalize(wi);
    pathSegment.throughput *= col;
}
