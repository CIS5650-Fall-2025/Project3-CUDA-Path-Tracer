#pragma once
#define JITTER 0.5
//#define DEBUG_NORMAL 1 // 1 : clamped, 0 : unclamped
//#define DEBUG_THROUGHPUT
//#define DEBUG_RADIANCE
 
//////////////////////////////////////////////////////////////////
/////////////////////Post Process Directives//////////////////////
//////////////////////////////////////////////////////////////////
 
#define POSTPROCESS
#define TONE_MAPPING_ACES 1
#define TONE_MAPPING_REINHARD 1
__device__ inline glm::vec3 ACESFilm(glm::vec3 x)
{
	return glm::clamp((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f);
}
__device__ inline glm::vec3 Reinhard(glm::vec3 x)
{
	return x / (x + glm::vec3(1.0f));
}

