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

    int meshStart;
    int meshEnd;

    int texIdx{ -1 };
    bool hasTexture{ false };
};

struct Texture {

    int width{ 0 };
    int height{ 0 };

    int channels;
    unsigned char* data;
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

    float R0sq;

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

// source: https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
struct BVHNode
{
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;

    int leftFirst;
    int triCount;

    int totalNodes;

    float cost() {
        return triCount * area();
    }

    void grow(const glm::vec3& p)
    {
        aabbMin = glm::min(aabbMin, p);
        aabbMax = glm::max(aabbMax, p);
    }

    float area()
    {
        glm::vec3 e = aabbMax - aabbMin;
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }
};

struct Bin { BVHNode bounds; int triCount = 0; };

struct Triangle
{
    glm::vec3 verts[3];
    glm::vec3 normals[3];
    glm::vec2 uvs[3];

    int geomIdx;

    // For each primitive stored in the BVH, 
    // we store the centroid of its bounding box
    glm::vec3 centroid;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    float t;
    glm::vec3 surfaceNormal;
    int materialId;

    int textureId{ -1 };
    glm::vec2 uv;
};