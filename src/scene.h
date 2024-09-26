#pragma once

#include <vector>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "intersections.h"

//using namespace std;

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
    Geom* geoms;
    Material* materials;
    uint32_t* materialIDs;
    Primitive* primitives;
    MTBVHNode* bvhNodes;
    AABB* bvhAABBs;
    glm::vec3* vertices;
    glm::vec3* normals;
    glm::vec2* uvs;

    // scene infos
    size_t bvhPitch;
    uint32_t bvhSize;
    uint32_t primNum;
    uint32_t triNum;

    void freeCudaMemory();
    __device__ inline bool intersect(const Ray& r, ShadeableIntersection& isect);
    __device__ inline bool intersectPrimitives(const Ray& r, uint32_t primID, float& dist);
    __device__ inline void intersectPrimitivesDetail(const Ray& r, uint32_t primID, ShadeableIntersection& isect);
};

class Scene
{
private:
    std::ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    bool loadObj(Object& newObj, const std::string& objPath);

public:
    Scene(std::string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Object> objects;
    std::unordered_map<std::string, uint32_t> MatNameToID;

    // full scene primitives
    std::vector<Primitive> primitives;
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> uvs;
    RenderState state;
    SceneDev* sceneDev;

    void buildDevSceneData();
};


__device__ inline int getMTBVHIdx(const glm::vec3 dir)
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

__device__ inline bool SceneDev::intersect(const Ray& r, ShadeableIntersection& isect)
{
    float tMin = FLT_MAX;
    uint32_t hitID = UINT32_MAX;
    int materialId = -1;

    int bvhIdx = getMTBVHIdx(r.direction);
    MTBVHNode* head = (MTBVHNode*)((char*)bvhNodes + bvhIdx * bvhPitch);
    uint32_t curr = 0;
    glm::vec3 invDir = 1.f / r.direction;

    while (curr != bvhSize) {
        bool hasHit = bvhAABBs[head[curr].bboxID].intersect(r.origin, invDir, tMin);

        if (hasHit)
        {
            uint32_t s = head[curr].startPrimID;
            uint32_t e = head[curr].endPrimID;
            if (s != UINT32_MAX)
            {
                for (uint32_t i = s; i < e; ++i)
                {
                    float dist;
                    uint32_t primId = primitives[i].primId;
                    int mid = primitives[i].materialId;
                    bool hit = intersectPrimitives(r, primId, dist);

                    if (hit && dist < tMin)
                    {
                        tMin = dist;
                        hitID = primId;
                        materialId = mid;
                    }
                }
            }
            ++curr;
        }
        else
        {
            curr = head[curr].missNext;
        }
    }

    isect.primId = hitID;
    if (hitID != UINT32_MAX)
    {
        isect.t = tMin;
        isect.materialId = materialId;
        intersectPrimitivesDetail(r, hitID, isect);
        return true;
    }
    else
    {
        isect.t = -1.f;
        return false;
    }
}

__device__ inline bool SceneDev::intersectPrimitives(const Ray& r, uint32_t primID, float& dist)
{
    if (primID < triNum)
    {
        glm::vec3 bary;
        float t = triangleIntersectionTest(r, vertices[3 * primID], vertices[3 * primID + 1], vertices[3 * primID + 2]);
        if (t < 0.f)
        {
            dist = -1.f;
            return false;
        }
        else
        {
            dist = t;
            return true;
        }
    }
    else
    {
        Geom& geom = geoms[primID - triNum];
        float t;
        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, r);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, r);
        }

        if (t < 0.f)
        {
            dist = -1.f;
            return false;
        }
        else
        {
            dist = t;
            return true;
        }

    }
    return false;
}

__device__ inline void SceneDev::intersectPrimitivesDetail(const Ray& r, uint32_t primID, ShadeableIntersection& isect)
{
    if (primID < triNum)
    {
        glm::vec3 tmpNor;
        glm::vec3 bary;
        triangleIntersection(r, vertices[3 * primID], vertices[3 * primID + 1], vertices[3 * primID + 2], tmpNor, bary);
        isect.nor = normals[3 * primID] * bary.x + normals[3 * primID + 1] * bary.y
            + normals[3 * primID + 2] * bary.z;
        isect.uv = uvs[3 * primID] * bary.x + uvs[3 * primID + 1] * bary.y
            + uvs[3 * primID + 2] * bary.z;

    }
    else
    {
        Geom& geom = geoms[primID - triNum];
        glm::vec3 isectPoint;
        float t;
        bool outside;
        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, r, isectPoint, isect.nor, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, r, isectPoint, isect.nor, outside);
        }

    }
}