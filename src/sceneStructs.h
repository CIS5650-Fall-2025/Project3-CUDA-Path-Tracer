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

enum MatType {
    DIFFUSE, 
    MIRROR, 
    DIELECTRIC, 
    MICROFACET
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle {
    glm::vec3 points[3];
    glm::vec3 planeNormal;
    glm::vec3 normals[3];
    glm::vec2 uvs[3];
    float cdf = 0.0f;

    Triangle() {}
    Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3)
    : points{p1, p2, p3},
      planeNormal(glm::normalize(glm::cross(p2 - p1, p3 - p2))),
      normals{planeNormal, planeNormal, planeNormal},
      uvs{glm::vec2(), glm::vec2(), glm::vec2()} {}
};

struct Geom
{
    enum GeomType type;
    int materialid;
    int numTriangles = 0;
    float area = 0.0f;

    Triangle* triangles = nullptr; // Host-side pointer
    Triangle* devTriangles = nullptr; // Device-side pointer

    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material
{
    int type;
    glm::vec3 color;
    float roughness;
    bool isSpecular = false;
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
    bool hasHitLight;
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
