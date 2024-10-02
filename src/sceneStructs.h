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
    OBJECT
};

struct Triangle {
    glm::vec3 vertices[3];
    glm::vec3 normals[3];
    glm::vec2 uv2[2];
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
    int triangleIndex;
    int triangleCount;
    glm::vec3 boundingBoxMin;
    glm::vec3 boundingBoxMax;
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
    float focus_distance;
    float lens_radius;
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


struct BoundingBox {
    glm::vec3 min;
    glm::vec3 max;

    BoundingBox() : min(glm::vec3(0.f)), max(glm::vec3(0.f)) {}

    BoundingBox(const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3)
    {
        min = glm::vec3(glm::min(p1.x, p2.x), glm::min(p1.y, p2.y), glm::min(p1.z, p2.z));
        max = glm::vec3(glm::max(p1.x, p2.x), glm::max(p1.y, p2.y), glm::max(p1.z, p2.z));

        min = glm::vec3(glm::min(p3.x, min.x), glm::min(p3.y, min.y), glm::min(p3.z, min.z));
        max = glm::vec3(glm::max(p3.x, max.x), glm::max(p3.y, max.y), glm::max(p3.z, max.z));
    }
};