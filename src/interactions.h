#pragma once

#include "intersections.h"
#include <glm/glm.hpp>
#include <thrust/random.h>

const __device__ __constant__ float INV_PI = 0.31830988618379067f;

const __device__ __constant__ float PI_OVER_FOUR = 0.78539816339744831f;

const __device__ __constant__ float PI_OVER_TWO = 1.57079632679489662f;

//const __device__ __constant__ float

//Utility Functions
inline __device__ float CosTheta(const glm::vec3& w) { return w.z; }
inline __device__ float AbsCosTheta(const glm::vec3& w) { return glm::abs(w.z); }

//BSDF Functions
inline __device__ void coordinateSystem(const glm::vec3 v1, glm::vec3& v2, glm::vec3& v3) {
    if (abs(v1.x) > abs(v1.y))
        v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = cross(v1, v2);
}

/**
* LocalToWorld returns a mat3 that transforms hemisphere local space (tangent spacce) vectors back to world space
* (0,0,1) is the up vector/normal vector in tangent space
**/
inline __device__ glm::mat3 LocalToWorld(glm::vec3 nor) {
    glm::vec3 tan, bit;
    //Obtain tan and bitangent using coordinateSystem(...)
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

/**
* WorldToLocal returns a mat3 that transforms world space vectors to normal aligned hemisphere space vectors
* (0,0,1) is the up vector/normal vector in tangent space
**/
inline __device__ glm::mat3 WorldToLocal(glm::vec3 nor) {
    return transpose(LocalToWorld(nor));
}

//////////////////////////////////////////////////////////////////////////////
/////////////////////////////SAMPLE WARPING///////////////////////////////////

/**
* Map random uv on a unit square ([0,1],[0,1]) -> to a point on a unit circle with origin (0,0)!
*/

__device__ void squareToDiskConcentric(const glm::vec2 xi, glm::vec3* wi);
//Malley's method
__device__ void squareToHemisphereCosine(const glm::vec2 xi, glm::vec3* wi);

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

///////////////////////////SAMPLE WARPING END/////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//DIFFUSE//
__device__ void f_diffuse(
    glm::vec3& f,
    const Material& m);

__device__ void pdf_diffuse(
    float& pdf, const glm::vec3& wi);

__device__ void sample_f_diffuse(
    PathSegment& pathSegment,
    float& pdf,
    glm::vec3& f,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng);
//DIFFUSE//

/**
* Given an incoming w_o, and an intersection, evaluate the BSDF to find:
*   f(), pdf() and wiW
**/
__device__ void sample_f(
    PathSegment& pathSegment,
    float& pdf,
    glm::vec3 &f,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 normal,
    thrust::default_random_engine& rng);
