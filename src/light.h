#pragma once
#include "utilities.h"
#include <thrust/random.h>

glm::vec3 DirectSampleAreaLight(
    int idx,
    glm::vec3 view_point, 
    glm::vec3 view_nor,
    int num_lights,
    glm::vec3& wiW, 
    float& pdf,
    thrust::default_random_engine& rng)
{
	return glm::vec3(1.0f);
}

glm::vec3 Sample_Li(
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