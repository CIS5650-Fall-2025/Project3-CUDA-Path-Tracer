#pragma once

#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "intersections.h"


// range: [startPrimID, endPrimID)
struct BVHNode
{
    AABB bbox;
    BVHNode* left;
    BVHNode* right;
    uint32_t startPrimID;
    uint32_t endPrimID;
    uint32_t nodeID;
};

struct MTBVHNode
{
    uint32_t bboxID;
    uint32_t missNext;
    uint32_t startPrimID;
    uint32_t endPrimID;
};

struct SceneDev
{
    // dev buffer handles
    Geom* geoms = nullptr;
    Material* materials = nullptr;
    PrimitiveDev* primitives = nullptr;
    MTBVHNode* bvhNodes = nullptr;
    AABB* bvhAABBs = nullptr;
    glm::vec3* vertices = nullptr;
    glm::vec3* normals = nullptr;
    glm::vec2* uvs = nullptr;

    // scene infos
    cudaTextureObject_t envMap = 0;
    EnvMapDistrib* envMapDistrib = nullptr;
    float envMapPdfSumInv;
    int envMapWidth;
    int envMapHeight;
    size_t bvhPitch;
    uint32_t bvhSize;
    uint32_t primNum;
    uint32_t triNum;

    void freeCudaMemory();
    __device__ bool intersect(const Ray& r, ShadeableIntersection& isect);
    __device__ bool intersectPrimitives(const Ray& r, uint32_t primID, float& dist);
    __device__ bool visibilityTest(const Ray& r, float dist);
    __device__ void intersectPrimitivesDetail(const Ray& r, uint32_t primID, ShadeableIntersection& isect);
    __device__  glm::vec3 sampleEnv(const glm::vec3& ori, glm::vec3& wi, glm::vec3 rng, float* pdf);
    __device__  glm::vec3 getEnvColor(const glm::vec3& dir);
    __device__  float envMapPdf(const glm::vec3& wi);
};


__device__ inline int getMTBVHIdx(const glm::vec3& dir)
{
    int idx = 0;
    float maxV = -1.f;
    for (int i = 0; i < 3; ++i)
    {
        if (glm::abs(dir[i]) > maxV)
        {
            idx = i;
            maxV = glm::abs(dir[i]);
        }
    }
    return 2 * idx + ((dir[idx] > 0.f) ? 0 : 1);
}


