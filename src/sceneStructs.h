#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

static constexpr float MachineEpsilon = std::numeric_limits<float>::epsilon() * 0.5;
__inline__ __host__ __device__ constexpr float gamma(int n) {
	return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
	MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct AABB
{
	glm::vec3 min;
	glm::vec3 max;
	AABB() : min(glm::vec3(FLT_MAX)), max(glm::vec3(FLT_MIN)) {}
	static AABB Union(const AABB& b1, const AABB& b2)
	{
		AABB ret;
		ret.min = glm::min(b1.min, b2.min);
		ret.max = glm::max(b1.max, b2.max);
		return ret;
	}

	static AABB Union(const AABB& b, const glm::vec3& p)
	{
		AABB ret;
		ret.min = glm::min(b.min, p);
		ret.max = glm::max(b.max, p);
		return ret;
	}

	int maxExtent() const
	{
		glm::vec3 diag = max - min;
		if (diag.x > diag.y && diag.x > diag.z)
			return 0;
		else if (diag.y > diag.z)
			return 1;
		else
			return 2;
	}

	__host__ __device__ glm::vec3 Offset(const glm::vec3& p) const
	{
		glm::vec3 o = p - min;
		if (max.x > min.x) o.x /= max.x - min.x;
		if (max.y > min.y) o.y /= max.y - min.y;
		if (max.z > min.z) o.z /= max.z - min.z;
		return o;
	}

	float SurfaceArea() const
	{
		glm::vec3 d = max - min;
		return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
	}

	__inline__ __device__ bool IntersectP(const Ray& ray, float* hitt0 = nullptr,
		float* hitt1 = nullptr) const {
		float t0 = 0, t1 = 2000;
		for (int i = 0; i < 3; ++i) {
			float invRayDir = 1 / ray.direction[i];
			float tNear = (min[i] - ray.origin[i]) * invRayDir;
			float tFar = (max[i] - ray.origin[i]) * invRayDir;
			// Update parametric interval from slab intersection  values
			if (tNear > tFar)
			{
				float temp = tNear;
				tNear = tFar;
				tFar = temp;
			}
			// Update tFar to ensure robust ray–bounds intersection 
			tFar *= 1 + 2 * gamma(3);

			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar < t1 ? tFar : t1;
			if (t0 > t1) return false;
		}
		if (hitt0) *hitt0 = t0;
		if (hitt1) *hitt1 = t1;
		return true;
	}

};

struct Triangle
{
	glm::vec3 vertices[3];
	glm::vec3 normals[3];
	glm::vec2 uvs[3];
	__device__ float intersect(const Ray& r) const
	{
		// Moller-Trumbore algorithm
		glm::vec3 e1 = vertices[1] - vertices[0];
		glm::vec3 e2 = vertices[2] - vertices[0];
		glm::vec3 s1 = glm::cross(r.direction, e2);
		float divisor = glm::dot(s1, e1);
		if (divisor == 0.0f)
		{
			return -1.0f;
		}
		float invDivisor = 1.0f / divisor;

		glm::vec3 d = r.origin - vertices[0];
		float b1 = glm::dot(d, s1) * invDivisor;
		if (b1 < 0.0f || b1 > 1.0f)
		{
			return -1.0f;
		}

		glm::vec3 s2 = glm::cross(d, e1);
		float b2 = glm::dot(r.direction, s2) * invDivisor;
		if (b2 < 0.0f || b1 + b2 > 1.0f)
		{
			return -1.0f;
		}

		float t = glm::dot(e2, s2) * invDivisor;

		return t;
	}

	__inline__ __device__ glm::vec3 getBarycentricCoordinates(glm::vec3 insectPoint) const
	{
		// Barycentric coordinates
		float u, v, w;
		glm::vec3 e1 = vertices[1] - vertices[0];
		glm::vec3 e2 = vertices[2] - vertices[0];
		glm::vec3 ei = insectPoint - vertices[0];
		float s = glm::length(glm::cross(e1, e2)) / 2.0;
		float s1 = glm::length(glm::cross(ei, e2)) / 2.0;
		float s2 = glm::length(glm::cross(e1, ei)) / 2.0;
		u = s1 / s;
		v = s2 / s;
		w = 1.0f - u - v;
		return glm::vec3(w, u, v);
	}

	__inline__ __device__ glm::vec3 getNormal() const
	{
		return glm::normalize(glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
	}
	__inline__ __device__ glm::vec3 getNormal(glm::vec3 insectPoint) const
	{
		glm::vec3 barycentric = getBarycentricCoordinates(insectPoint);
		return barycentric.x * normals[0] + barycentric.y * normals[1] + barycentric.z * normals[2];
	}

	__inline__ __device__ glm::vec2 getUV(glm::vec3 insectPoint) const
	{
		glm::vec3 barycentric = getBarycentricCoordinates(insectPoint);
		return barycentric.x * uvs[0] + barycentric.y * uvs[1] + barycentric.z * uvs[2];
	}

	__inline__ __device__ glm::vec3 getCenter() const
	{
		return (vertices[0] + vertices[1] + vertices[2]) / 3.0f;
	}

	__inline__ __host__ AABB getBounds() const
	{
		AABB aabb;
		aabb.min = glm::min(vertices[0], glm::min(vertices[1], vertices[2]));
		aabb.max = glm::max(vertices[0], glm::max(vertices[1], vertices[2]));
		return aabb;
	}

	int materialid;
};

struct Geom
{
	Geom() : type(MESH), materialid(-1), translation(glm::vec3(0.0f)), rotation(glm::vec3(0.0f)), scale(glm::vec3(1.0f)), triangleStartIdx(0), triangleEndIdx(0) {}
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

	int triangleStartIdx;
	int triangleEndIdx;
	int getNumTriangles() const { return triangleEndIdx - triangleStartIdx; }
};

struct Material
{
	Material() : color(glm::vec3(0.0f)), hasReflective(0.0f), hasRefractive(0.0f), ior(1.0f), emittance(0.0f), roughness(0.0f), metallic(0.0f), materialId(-1) {}
	Material(glm::vec3 col) : color(col), hasReflective(0.0f), hasRefractive(0.0f), ior(1.0f), emittance(0.0f), roughness(0.0f), metallic(0.0f), materialId(-1) {}
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
	float reflective;
	float refractive;
    float ior;
    float emittance;
	float roughness;
	float metallic;
	int materialId;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct alignas(64) PathSegment
{
    Ray ray;
    glm::vec3 color;
	glm::vec3 throughput;
	int pixelIndex;
	int remainingBounces;

    __host__ __device__ bool isTerminated() const {
        return remainingBounces <= 0;
    }
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv;
};
