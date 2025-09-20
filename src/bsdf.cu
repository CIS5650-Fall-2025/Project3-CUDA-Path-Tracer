#include "bsdf.h"
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "utilities.h"
#include "intersections.h"
#include <glm/gtx/norm.hpp>

#include "ggx.h"

__global__ void shade(int depth,
                      int iter,
                      int numPaths,
                      const ShadeableIntersection* shadeableIntersections,
                      const Material* materials,
                      PathSegment* pathSegments)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numPaths)
	{
		return;
	}

    auto& pathSegment = pathSegments[idx];
    auto intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f)
    {
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

		auto xi = glm::vec3(u01(rng), u01(rng), u01(rng));

		// This is basically just one material type: like principled BSDF in Blender
		// Cool stuff like glass needs glTF extensions like transmission
        
        const auto material = materials[intersection.materialId];
        // Every object has uniform material. But if you hit the part of the object that is not emissive, you would want to continue
        // Also with MIS we would somehow need to get a PDF, which is not possible like this
        if (glm::length2(material.emissive) > 0.f)
        {
            pathSegment.color *= material.emissive;
            pathSegment.remainingBounces = 0;
        }
        else
        {
            IntersectionData isect;
            isect.position = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
            isect.normal = intersection.surfaceNormal;

            glm::vec3 wi;
            float pdf;
            auto brdf = sampleF(material, isect, -pathSegment.ray.direction, xi, &wi, &pdf);
            float NdotL = glm::max(0.f, glm::dot(isect.normal, wi)); // Use wi after sampling

            pathSegment.ray = spawnRay(isect.position, wi);
            pathSegment.remainingBounces--; // This should be ok since only one thread writes to this path segment

            if (pdf != 0)
            {
                pathSegment.color *= brdf * NdotL / pdf;
            }
        }
    }
    else 
    {
        pathSegment.remainingBounces = 0;
    }
}

// Sampling and coordinate conversion adapted from my CIS 5610 path tracer

__device__ glm::vec3 squareToHemisphereCosine(const glm::vec2& xi) 
{
    float r = glm::sqrt(xi.x);
    float theta = 2.0 * PI * xi.y;
    glm::vec2 disk = glm::vec2(r * cos(theta), r * sin(theta));
    return glm::vec3(disk.x, disk.y, glm::max(0.f, glm::sqrt(1.f - dot(disk, disk))));
}
__device__ float squareToHemisphereCosinePDF(const glm::vec3& sample)
{
    return glm::abs(sample.z) / PI;
}

__device__ void coordinateSystem(const glm::vec3& v1, glm::vec3* v2, glm::vec3* v3)
{
    if (abs(v1.x) > abs(v1.y))
    {
        *v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    }
    else
    {
        *v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    }
    *v3 = glm::cross(v1, *v2);
}
__device__ glm::mat3 LocalToWorld(const glm::vec3& nor)
{
    glm::vec3 tan, bit;
    coordinateSystem(nor, &tan, &bit);
    return glm::mat3(tan, bit, nor);
}
__device__ glm::mat3 WorldToLocal(const glm::vec3& nor)
{
    return transpose(LocalToWorld(nor));
}

// GGX model as described in https://seblagarde.wordpress.com/wp-content/uploads/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf

__device__ glm::vec3 F_Schlick(glm::vec3 f0, float f90, float u)
{
    return f0 + (f90 - f0) * glm::pow(1.f - u, 5.f);
}

__device__ float Fr_DisneyDiffuse(float NdotV, float NdotL, float LdotH, float linearRoughness)
{
    float energyBias = glm::mix(0.f, 0.5f, linearRoughness);
    float energyFactor = glm::mix(1.0f, 1.0f / 1.51f, linearRoughness);
    float fd90 = energyBias + 2.0f * LdotH*LdotH * linearRoughness;
    glm::vec3 f0 = glm::vec3(1.0f);
    float lightScatter = F_Schlick(f0, fd90, NdotL).r;
    float viewScatter = F_Schlick(f0, fd90, NdotV).r;

    return lightScatter * viewScatter * energyFactor;
}

__device__ float V_SmithGGXCorrelated(float NdotL, float NdotV, float alphaG)
{
    float alphaG2 = alphaG * alphaG;
    alphaG2 = alphaG2 + 0.0000001f;
    float Lambda_GGXV = NdotL * sqrt((-NdotV * alphaG2 + NdotV) * NdotV + alphaG2);
    float Lambda_GGXL = NdotV * sqrt((-NdotL * alphaG2 + NdotL) * NdotL + alphaG2);

    return 0.5f / (Lambda_GGXV + Lambda_GGXL);
}

__device__ float D_GGX(float NdotH, float m)
{
    float m2 = m * m;
    float f = (NdotH * m2 - NdotH) * NdotH + 1;
    return m2 / (f * f) / PI;
}

__device__ glm::vec3 GetF(float LdotH, glm::vec3 f0)
{
    float f90 = glm::clamp(50.0f * glm::dot(f0, glm::vec3(0.33f)), 0.0f, 1.0f);
    return F_Schlick(f0, f90, LdotH);
}

__device__ glm::vec3 GetSpecular(float NdotV, float NdotL, float LdotH, float NdotH, float roughness, glm::vec3 f0, glm::vec3* F)
{
    *F = GetF(LdotH, f0);
    float Vis = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
    float D = D_GGX(NdotH, roughness);
    glm::vec3 Fr = D * (*F) * Vis / PI;
    return Fr;
}

__device__ glm::vec3 sampleF(const Material& material, const IntersectionData &isect, const glm::vec3& woW, const glm::vec3& xi, glm::vec3* wiW, float* pdf)
{
    glm::vec3 N = isect.normal;
    glm::vec3 V = woW; // In rasterizer you would take vector from fragment position

    const glm::vec3 albedo = glm::vec3(material.albedo);
    const float roughness = material.roughness;
    const float metallic = material.metallic;

    float ggx_pdf = 0;
    float diffuse_pdf = 0;
    if (xi.z < metallic)
    {
        if (roughness == 0)
        {
	        *wiW = reflect(-woW, N);
            ggx_pdf = 1.f;
            diffuse_pdf = 0.f;
        }
        else
        {
            auto wo = WorldToLocal(N) * woW;
            if (wo.z == 0) return glm::vec3(0.);

            glm::vec3 wh = Sample_wh(wo, glm::vec2(xi), roughness);
            glm::vec3 wi = reflect(-wo, wh);
            diffuse_pdf = squareToHemisphereCosinePDF(*wiW);
            *wiW = LocalToWorld(N) * wi;
            if (!SameHemisphere(wo, wi)) return glm::vec3(0.f);
            ggx_pdf = TrowbridgeReitzPdf(wo, wh, roughness) / (4 * dot(wo, wh));
        }
    }
    else
    {
        // Diffuse Sample
        *wiW = squareToHemisphereCosine(glm::vec2(xi));
        // TODO: I think ggx_pdf is wrong here but it looks ok I guess
        diffuse_pdf = squareToHemisphereCosinePDF(*wiW);
        *wiW = LocalToWorld(N) * *wiW;
    }

	*pdf = material.metallic * ggx_pdf + (1.f - material.metallic) * diffuse_pdf;

	const auto& L = *wiW;

    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, glm::vec3(metallic));

    const glm::vec3	H = glm::normalize(L + V);	
    const float	NdotV = abs(dot(N, V)) + 1e-5f;
    const float NdotL = glm::max(0.f, dot(L, N));
	const float LdotH = glm::max(0.f, dot(L, H));
    const float	NdotH = glm::max(0.f, dot(H, N));

    glm::vec3 F;
    glm::vec3 specular = GetSpecular(NdotV, NdotL, LdotH, NdotH, roughness, F0, &F);

    // Otherwise dividing by roughness in GGX which results in a NaN
    if (roughness == 0.f) 
    {
        float NdotL_spec = glm::max(0.f, dot(N, *wiW));
        if (NdotL_spec > 0.f) 
        {
            specular = glm::vec3(1.0f) / NdotL_spec;
        }
    	else 
        {
            specular = glm::vec3(0.f);
        }
    }

    float Fd = Fr_DisneyDiffuse(NdotV, NdotL, LdotH, roughness);
    glm::vec3 kD = (glm::vec3(1.f) - F) * (1.f - metallic);
    glm::vec3 diffuse = kD * (albedo * Fd / PI);

    return diffuse + specular;
}
