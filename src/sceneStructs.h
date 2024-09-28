#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

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

	__inline__ __device__ glm::vec3 getNormal() const
	{
		return glm::normalize(glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
	}

	__inline__ __device__ glm::vec3 getCenter() const
	{
		return (vertices[0] + vertices[1] + vertices[2]) / 3.0f;
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
	Material() : color(glm::vec3(0.5f)), hasReflective(0.0f), hasRefractive(0.0f), indexOfRefraction(1.0f), emittance(0.0f) {}
	Material(glm::vec3 col) : color(col), hasReflective(0.0f), hasRefractive(0.0f), indexOfRefraction(1.0f), emittance(0.0f) {}
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
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

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
	glm::vec3 throughput;

    __host__ __device__ void reset() {
		color = glm::vec3(0.0f);
		throughput = glm::vec3(1.0f);
	}
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
};
