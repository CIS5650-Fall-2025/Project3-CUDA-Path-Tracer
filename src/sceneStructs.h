#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "bbox.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define USE_BVH 1

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLE,
};

enum MatType
{
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    EMISSIVE
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
    glm::vec3 vertices[3];
    glm::vec3 normals[3];
    glm::vec2 uv[3];
    int numVertices;

    BBox bbox() {
        BBox bbox;
        for (int i = 0; i < numVertices; i++) {
            bbox.enclose(vertices[i]);
        }
        bbox.transform(transform);
        return bbox;
    }
};

struct Material
{
    enum MatType type;
    glm::vec3 color;
    float roughness;
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
    float focalLength;
    float apertureSize;
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
  int materialId; // materialId == -1 means no intersection
};
