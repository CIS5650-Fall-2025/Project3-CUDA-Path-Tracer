#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "sceneStructs.h"

struct IntersectionData
{
    glm::vec3 position;
    glm::vec3 normal;
};

// Megakernel using BRDF
__global__ void shade(int depth,
                      int iter,
                      int numPaths,
                      const ShadeableIntersection* shadeableIntersections, const Material* materials, PathSegment* pathSegments);

__device__ glm::vec3 sampleF(const Material& material, const IntersectionData &isect, const glm::vec3& woW, const glm::vec3& xi, glm::vec3* wiW, float* pdf);
