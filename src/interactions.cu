#include "interactions.h"
//#include "pbr.h"
#include "disneybsdf.h"
#include "light.h"

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



__device__ void scatterRay(
    PathSegment& pathSegment,
	const ShadeableIntersection& intersection,
    const glm::vec3& intersect,
    const Material &m,
    thrust::default_random_engine &rng,
    int num_lights,
    LinearBVHNode* dev_nodes,
    Triangle* dev_triangles,
    Light* dev_lights,
    cudaTextureObject_t envMap)
{
    
	glm::vec3 normal = intersection.surfaceNormal;
	glm::vec2 uv = intersection.uv;

	glm::vec3 wi = glm::vec3(0.0f);
	glm::vec3 col = glm::vec3(0.0f);
	Material mat = m;

    // TODO: implement PBR model
    thrust::uniform_real_distribution<float> u01(0, 1);
	glm::vec2 xi = glm::vec2(u01(rng), u01(rng));
	glm::mat3 ltw = LocalToWorld(normal);
	glm::mat3 wtl = glm::transpose(ltw);
    
	//col = Sample_disneyBSDF(m, pathSegment.ray, normal, xi, wi, ltw, wtl);

	// direct lighting
	float pdf_direct = 0.f;
	int light_id = 0;
    glm::vec3 Li_direct = Sample_Li(intersect, normal, wi, pdf_direct, intersection.directLightId, num_lights, envMap, rng, dev_nodes, dev_triangles, dev_lights, ltw, wtl);
 //   glm::vec3 Li_direct(m.color);
	col = Li_direct * AbsDot(wi, normal) / pdf_direct * m.color;

	//wi += glm::reflect(pathSegment.ray.direction, normal);

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
    //pathSegment.color = glm::vec3(Li_direct);
    //col = glm::vec3(1.f);
    //pathSegment.remainingBounces = 0;

	pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = glm::normalize(wi);
    pathSegment.throughput *= col;
}
