#pragma once
#include "utilities.h"
#include <thrust/random.h>
#include "sceneStructs.h"

// create enum class for light types
enum class LightType
{
	POINT,
	AREA,
	DIRECTIONAL
};


struct Light
{
	glm::vec3 color;
	glm::mat4 transform;
	glm::mat4 inverseTransform;
	LightType type;
};

extern Light* lights;
extern Light* dev_lights;

Light* lights = NULL;
Light* dev_lights = NULL;

__device__ glm::vec3 DirectSampleAreaLight(
    int idx,
    glm::vec3 view_point, 
    glm::vec3 view_nor,
    int num_lights,
    glm::vec3& wiW, 
    float& pdf,
    thrust::default_random_engine& rng)
{
	Light light = dev_lights[idx];
	glm::vec3 color = light.color;
    Ray shadowRay;

	// sample a point on the light
	glm::vec3 xi = glm::vec3(thrust::uniform_real_distribution<float>(-1.0f, 1.0f)(rng), thrust::uniform_real_distribution<float>(-1.0f, 1.0f)(rng), 0.0f);
	glm::vec4 vpl = light.inverseTransform * glm::vec4(xi, 1.0f);
	glm::vec3 wi = xi - glm::vec3(vpl);

	// compute cosTheta and distance
	float cosTheta = glm::dot(glm::vec3(0, 0, 1), normalize(-wi));
	wiW = glm::vec3(light.transform * glm::vec4(wi, 0.));
	float r = length(wiW);

	// compute pdf da
	pdf = r * r / (cosTheta * light.transform[2][0] * light.transform[2][1] * light.transform[2][2]);
	wiW = normalize(wiW);

	// check if there is block
	//shadowRay = SpawnRay(view_point, wiW);

	return glm::vec3(1.0f);
}

__device__ glm::vec3 Sample_Li(
    glm::vec3 view_point,
    glm::vec3 nor,
    glm::vec3& wiW,
    float& pdf,
    int* chosenLightIdx,
    int N_LIGHTS,
    cudaTextureObject_t ENV_MAP,
    thrust::default_random_engine& rng)
{
    // Choose a random light from among all of the
    // light sources in the scene, including the environment light
    int num_lights = N_LIGHTS;
	if (ENV_MAP != NULL)
	{
		num_lights++;
	}

    // choose a random light
	int randomLightIdx = thrust::uniform_int_distribution<int>(0, num_lights - 1)(rng);
	*chosenLightIdx = randomLightIdx;

    // choose an area light
	return DirectSampleAreaLight(randomLightIdx, view_point, nor, N_LIGHTS, wiW, pdf, rng);
}