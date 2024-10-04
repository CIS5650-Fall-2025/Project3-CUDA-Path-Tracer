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
    OBJ
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material
{
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



struct AABB
{
    glm::vec3 min;
    glm::vec3 max;

    // Default constructor initializes min to the largest possible value and max to the smallest.
    __host__ __device__ AABB() : min(glm::vec3(std::numeric_limits<float>::max())), max(glm::vec3(std::numeric_limits<float>::lowest())) {}

    // Constructor that initializes the AABB with given min and max points.
    __host__ __device__ AABB(const glm::vec3& min, const glm::vec3& max) : min(min), max(max) {}

    // Constructor that initializes the AABB to enclose three points.
    __host__ __device__ AABB(const glm::vec3& point1, const glm::vec3& point2, const glm::vec3& point3) {
        min = glm::min(point1, glm::min(point2, point3));
        max = glm::max(point1, glm::max(point2, point3));
    }

    // Method to find the index of the longest axis (0: x, 1: y, 2: z).
    __host__ __device__ int LongestAxisIndex() const {
        glm::vec3 diagonal = (max - min);
        int maxAxis = (diagonal.x > diagonal.y) ? 0 : 1;
        return (diagonal[maxAxis] > diagonal.z) ? maxAxis : 2;
    }

    // Method to compute the surface area of the AABB.
    __host__ __device__ float SurfaceArea() const {
        glm::vec3 d = (max - min);
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
};


struct Triangle
{
    glm::vec3 vertices[3];
    glm::vec3 normals[3];
    glm::vec2 uvs[3];

    AABB aabb = AABB();

    int idx_v0, idx_v1, idx_v2; // Indices of the vertices in the vertex array
};



struct Vertex {
    glm::vec3 pos;
    glm::vec3 nor;
};

