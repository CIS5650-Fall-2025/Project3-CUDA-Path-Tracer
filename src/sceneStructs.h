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

    Triangle(): 
        points{glm::vec3(), glm::vec3(), glm::vec3()},
        planeNormal(glm::vec3()),
        normals{glm::vec3(), glm::vec3(), glm::vec3()},
        uvs{glm::vec2(), glm::vec2(), glm::vec2()} {}

    Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 n1, glm::vec3 n2, glm::vec3 n3): 
        points{p1, p2, p3},
        planeNormal(glm::normalize(glm::cross(p2 - p1, p3 - p2))),
        normals{n1, n2, n3},
        uvs{glm::vec2(), glm::vec2(), glm::vec2()} {}

    Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3): 
        points{p1, p2, p3},
        planeNormal(glm::normalize(glm::cross(p2 - p1, p3 - p2))),
        normals{planeNormal, planeNormal, planeNormal},
        uvs{glm::vec2(), glm::vec2(), glm::vec2()} {}
};

/****** For BVH ******/
struct BoundingBox {
    glm::vec3 bMinCoors;
    glm::vec3 bMaxCoors;
    glm::vec3 center;
    glm::vec3 size;
    bool hasPoint;

    BoundingBox(): 
        bMinCoors(glm::vec3()), 
        bMaxCoors(glm::vec3()), 
        center(glm::vec3()), 
        size(glm::vec3()), 
        hasPoint(false) {}

    void resize(glm::vec3 min, glm::vec3 max) {
        if (hasPoint)
        {
            float minX = std::min(min.x, bMinCoors.x);
            float minY = std::min(min.y, bMinCoors.y);
            float minZ = std::min(min.z, bMinCoors.z);
            bMinCoors = glm::vec3(minX, minY, minZ);

            float maxX = std::max(max.x, bMaxCoors.x);
            float maxY = std::max(max.y, bMaxCoors.y);
            float maxZ = std::max(max.z, bMaxCoors.z);
            bMaxCoors = glm::vec3(maxX, maxY, maxZ);
        }
        else
        {
            hasPoint = true;
            bMinCoors = min;
            bMaxCoors = max;
        }

        // Update size and center based on the new min and max coordinates
        size = bMaxCoors - bMinCoors;
        center = (bMinCoors + bMaxCoors) / 2.0f;
    }
};

struct BVHTriangle {
    glm::vec3 center;
    glm::vec3 triMinCoors;
    glm::vec3 triMaxCoors;
    int index;

    BVHTriangle(): 
        center(glm::vec3()), 
        triMinCoors(glm::vec3()), 
        triMaxCoors(glm::vec3()), 
        index(0) {}

    BVHTriangle(glm::vec3 c, glm::vec3 min, glm::vec3 max, int idx) : 
        center(c), 
        triMinCoors(min), 
        triMaxCoors(max), 
        index(idx) {}
};

struct BVHNode
{
    glm::vec3 nMinCoors;
    glm::vec3 nMaxCoors;
    int startIdx;
    int numOfTris;

    BVHNode(): 
        nMinCoors(glm::vec3()), 
        nMaxCoors(glm::vec3()), 
        startIdx(0), 
        numOfTris(0) {};

    BVHNode(BoundingBox bbox): 
        nMinCoors(bbox.bMinCoors), 
        nMaxCoors(bbox.bMaxCoors), 
        startIdx(-1), 
        numOfTris(-1) {}

    BVHNode(BoundingBox bbox, int idx, int triangles): 
        nMinCoors(bbox.bMinCoors), 
        nMaxCoors(bbox.bMaxCoors), 
        startIdx(idx), 
        numOfTris(triangles) {}

    glm::vec3 computeBboxSize() { return nMaxCoors - nMinCoors; }
    glm::vec3 computeBboxCenter() { return (nMinCoors + nMaxCoors) * 0.5f; }
};
/*****************************************************************************************************************************/

struct Geom
{
    enum GeomType type;
    int materialid;
    int numTriangles = 0;
    float area = 0.0f;

    Triangle* triangles = nullptr; // Host-side pointer
    Triangle* devTriangles = nullptr; // Device-side pointer
    Triangle* bvhTriangles = nullptr; // Host-side pointer
    Triangle* devBvhTriangles = nullptr; // Device-side pointer

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
    glm::vec3 specularColor;
    float roughness;
    float emittance;
    float indexOfRefraction;
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
    float eta; // Used for Ruassin roulette to determine how likely this ray survives
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
