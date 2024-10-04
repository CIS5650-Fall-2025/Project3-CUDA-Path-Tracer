#pragma once

#include "intersections.h"
#include <glm/glm.hpp>
#include <thrust/random.h>
// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

// declare the kernel function that scatters rays based on material types
__host__ __device__ void scatter(const glm::vec3 intersection_color,
                                 const glm::vec3 intersection_point,
                                 const glm::vec3 intersection_normal,
                                 const Material intersection_material,
                                 thrust::default_random_engine generator,
                                 PathSegment& pathSegment);
