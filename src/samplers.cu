#include "samplers.h"

#include "utilities.h"

__host__ __device__ glm::vec3 randomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
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

__host__ __device__ glm::vec3 randomOnUnitSphere(thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float phi = u01(rng) * (2 * PI);
    float cosTheta = 2 * u01(rng) - 1;
    float sinTheta = sqrt(1 - cosTheta * cosTheta);

    return glm::vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

__host__ __device__ glm::vec2 randomOnUnitCircle(thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    // Rejection sampling for a unit circle of radius 1
    glm::vec2 p;
    glm::vec2 middle = glm::vec2(1.f);
    do
    {
        float xi1 = u01(rng) * 2.f;
        float xi2 = u01(rng) * 2.f;
        p = glm::vec2(xi1, xi2) - middle;
    } while (glm::length(p) >= 1.f);

    return p;
}

__host__ __device__ glm::vec4 sampleBilinear(ImageTextureInfo imageTextureInfo, glm::vec2 uv, glm::vec4* textures) {
    
	if (imageTextureInfo.width < 1 || imageTextureInfo.height < 1) {
		return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	}

	// Scale texture coordinates to image space
	float x = imageTextureInfo.width * uv.x;
	float y = imageTextureInfo.height * uv.y;

	// Shift so texel centers are at integer locations
	x -= 0.5f;
	y -= 0.5f;

    // Find 4 bounding texel coordinates
	int32_t ix = int32_t(floor(x));
	int32_t iy = int32_t(floor(y));
	int32_t ix0 = min(max(ix,  0),int32_t(imageTextureInfo.width)-1);
	int32_t ix1 = min(max(ix+1,0),int32_t(imageTextureInfo.width)-1);
	int32_t iy0 = min(max(iy,  0),int32_t(imageTextureInfo.height)-1);
	int32_t iy1 = min(max(iy+1,0),int32_t(imageTextureInfo.height)-1);
	glm::vec4 s00 = textures[imageTextureInfo.index + iy0 * imageTextureInfo.width + ix0];
	glm::vec4 s10 = textures[imageTextureInfo.index + iy0 * imageTextureInfo.width + ix1];
	glm::vec4 s01 = textures[imageTextureInfo.index + iy1 * imageTextureInfo.width + ix0];
	glm::vec4 s11 = textures[imageTextureInfo.index + iy1 * imageTextureInfo.width + ix1];

	// Bilinearly interpolate
	glm::vec4 s0 = (y - iy) * (s01 - s00) + s00;
	glm::vec4 s1 = (y - iy) * (s11 - s10) + s10;
	return (x - ix) * (s1 - s0) + s0;
}