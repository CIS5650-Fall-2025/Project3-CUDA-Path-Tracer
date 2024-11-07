#include "interactions.h"

#define TEST_DIFFUSE 0
#define USE_DIFFUSE_TEXTURE 1
#define USE_NORMAL_TEXTURE 1
#define TEST_NORMAL_MAP 0

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

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

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ glm::vec3 sampleGGX(const glm::vec3& n, float roughness, thrust::default_random_engine& rng) {
    float alpha = roughness * roughness;  // Square of roughness to use in the distribution

    // Sample random angles
    thrust::uniform_real_distribution<float> u01(0, 1);
    float phi = 2.0 * PI * u01(rng);
    float cosTheta = sqrt((1.0 - u01(rng)) / (1.0 + (alpha * alpha - 1.0) * u01(rng)));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // Spherical to Cartesian coordinates
    glm::vec3 h;
    h.x = cos(phi) * sinTheta;
    h.y = sin(phi) * sinTheta;
    h.z = cosTheta;

    // Transform h to align with the surface normal
    glm::vec3 upVector = abs(n.z) < 0.999 ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
    glm::vec3 tangentX = glm::normalize(glm::cross(upVector, n));
    glm::vec3 tangentY = glm::cross(n, tangentX);

    // Transform h into the tangent space of the normal n
    return tangentX * h.x + tangentY * h.y + n * h.z;
}

__host__ __device__ glm::vec3 textureSample(const Texture* texture, glm::vec2 uv) {
    int x = (int)(uv.x * texture->width);
    int y = (int)((1.0f - uv.y) * texture->height);
    int index = 0;
    index = y * texture->width + x;
    glm::vec3 texColor = texture->data[index];
    return texColor;
}

__host__ __device__ float GGXDistribution(const glm::vec3& n, const glm::vec3& h, float roughness) {
    float alpha = roughness * roughness;  // Roughness squared
    float alpha2 = alpha * alpha;
    float cosThetaH = glm::dot(n, h);
    float cosThetaH2 = cosThetaH * cosThetaH;

    float denom = cosThetaH2 * (alpha2 - 1.0) + 1.0;
    denom = PI * denom * denom;

    return alpha2 / denom;
}

__host__ __device__ float GeometrySmith(const glm::vec3& n, const glm::vec3& v, const glm::vec3& l, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float cosThetaV = glm::dot(n, v);
    float cosThetaL = glm::dot(n, l);

    float twoTanThetaV = abs(2.0 * cosThetaV / sqrt(1.0 - cosThetaV * cosThetaV));
    float lambdaV = (-1.0 + sqrt(1.0 + alpha2 * twoTanThetaV * twoTanThetaV)) / 2.0;

    float twoTanThetaL = abs(2.0 * cosThetaL / sqrt(1.0 - cosThetaL * cosThetaL));
    float lambdaL = (-1.0 + sqrt(1.0 + alpha2 * twoTanThetaL * twoTanThetaL)) / 2.0;

    return 1.0 / ((1.0 + lambdaV) * (1.0 + lambdaL));
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3 tangent,
    glm::vec3 bitangent,
    glm::vec2 uv,
    //glm::mat3 TBN,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // Set the new ray origin and direction to the intersection point
	glm::vec3 newOrigin = intersect + EPSILON * normal;   
    glm::vec3 newDirection = normalize(calculateRandomDirectionInHemisphere(normal, rng));

    float pdf = 1.0;

#if !TEST_DIFFUSE
    // Diffuse shading (Lambertian reflection)
    if (!m.hasReflective && !m.hasRefractive && !m.hasDiffuseTexture) {
        // Simple diffuse scattering
        pathSegment.ray.direction = newDirection;

		float cosTheta = glm::abs(glm::dot(normal, newDirection));
		pdf = cosTheta / PI;

        // throughput
        if (pdf > EPSILON) {
            pathSegment.color *= m.color;
        }
    }
    else if (m.hasReflective) {
#if 0
        // Reflective materials: Specular reflection
        glm::vec3 incoming = pathSegment.ray.direction;
        glm::vec3 reflected;

        glm::vec3 h = sampleGGX(normal, m.specular.roughness, rng);
        reflected = glm::reflect(-incoming, h);

        float cosTheta = glm::dot(incoming, h);
        glm::vec3 F0 = glm::vec3(0.04);  // Assuming non-metallic; adjust if metallic
        glm::vec3 fresnel = F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);

        float D = GGXDistribution(normal, h, m.specular.roughness);
        float G = GeometrySmith(normal, incoming, reflected, m.specular.roughness);
        glm::vec3 specColor = (D * fresnel * G) / (4.0f * glm::dot(normal, incoming) * glm::dot(normal, reflected));

        // Adjust the color for reflective materials using the specular component
        pathSegment.color *= m.specular.color * specColor;
        newDirection = reflected;
#endif
        // Reflective materials: Specular reflection
        newDirection = glm::reflect(pathSegment.ray.direction, normal);
        // calculate specular reflection color

        // Adjust the color for reflective materials using the specular component
        pathSegment.color *= m.color;
    }
    else if (m.hasRefractive) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        float rand = u01(rng);
        float cosTheta = glm::dot(normal, pathSegment.ray.direction);
        float eta = (cosTheta > 0) ? (m.indexOfRefraction / 1.0f) : (1.0f / m.indexOfRefraction);
        glm::vec3 refractDirection = glm::normalize( glm::refract(pathSegment.ray.direction, normal, eta));

        // Adjust normal direction and cosTheta for refraction calculations
        if (cosTheta < 0) {
            cosTheta = -cosTheta; //entering the medium
        }
        else {
            normal = -normal; // Flip the normal
        }

        // Calculate Fresnel reflectance using Schlick's approximation
        float R0 = pow((1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction), 2);
        float reflectance = R0 + (1 - R0) * pow(1 - cosTheta, 5);

        // Check if the refraction results in total internal reflection
        if (glm::length(refractDirection) == 0) {
            reflectance = 1.0; 
        }

        if (rand < reflectance) {
            // Reflect
            newDirection = glm::reflect(pathSegment.ray.direction, normal);
        }
        else {
            // Refract
            newDirection = refractDirection;
        }
		newOrigin = intersect - EPSILON * normal;
        pathSegment.color *= m.color;      
    }

    // texture
#if USE_DIFFUSE_TEXTURE
    if (m.hasDiffuseTexture) {
        glm::vec3 textureColor = textureSample(m.diffuseTexture, uv);
        // gamma correction to texture color
        //textureColor = glm::pow(textureColor, glm::vec3(1.0f / 2.2f));
        pathSegment.color *= textureColor;
    }
#endif
#if USE_NORMAL_TEXTURE
    if (m.hasNormalTexture) {
        glm::vec3 sampledNormal = textureSample(m.normalTexture, uv);
        sampledNormal = 2.0f * sampledNormal - glm::vec3(1.0f); // [0,1] to [-1, 1]
		glm::mat3 TBN = glm::mat3(tangent, bitangent, normal);
        normal = normalize(TBN * sampledNormal);
        newOrigin = intersect;
        newDirection = normalize(calculateRandomDirectionInHemisphere(normal, rng));
#if TEST_NORMAL_MAP
        pathSegment.color = sampledNormal;
#endif
    }
#endif

#else   
    //test diffuse
	

	float cosTheta = glm::dot(normal, newDirection);
	pdf = cosTheta * INV_PI;

	// Multiply the path color by the material color (throughput)
	if (pdf > EPSILON) {
		pathSegment.color *= m.color; // why not divide by PI?
	}
    else {
		pathSegment.color = glm::vec3(0.0f);
        return;
    }

#endif

	// UPDATE the path segment with the new ray origin and direction
    pathSegment.ray.origin = newOrigin;
    pathSegment.ray.direction = newDirection;
    pathSegment.remainingBounces--;
}

__host__ __device__ void getEnvironmentMapColor(
    PathSegment& pathSegment,
    const Texture& enviromentMap,
    thrust::default_random_engine& rng) {

	glm::vec3 rayDir = -pathSegment.ray.direction;
    rayDir = glm::normalize(rayDir);
    float u = 0.5f + (atan2(rayDir.z, rayDir.x) / (2.0f * PI));
    float v = 0.5f - (asin(rayDir.y) / PI);
    glm::vec2 uv(u, v);

	// sample the environment map
	glm::vec3 color = textureSample(&enviromentMap, uv);
	pathSegment.color *= color;

}
