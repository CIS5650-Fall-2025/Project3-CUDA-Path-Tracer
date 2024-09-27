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
    size_t bvhPitch;
    uint32_t bvhSize;
    uint32_t primNum;
    uint32_t triNum;

    void freeCudaMemory();
    __device__ inline bool intersect(const Ray& r, ShadeableIntersection& isect);
    __device__ inline bool intersectPrimitives(const Ray& r, uint32_t primID, float& dist);
    __device__ inline void intersectPrimitivesDetail(const Ray& r, uint32_t primID, ShadeableIntersection& isect);
    __device__ inline void sampleEnv(PathSegment& segment);
};

class Scene
{
private:
    std::ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    bool loadObj(Object& newObj, const std::string& objPath);
    void loadTextureFile(const std::string& texPath, cudaTextureObject_t& texObj);
    void createCudaTexture(void* data, int width, int height, cudaTextureObject_t& texObj, bool isHDR);

    void scatterPrimitives(std::vector<Primitive>& srcPrim, std::vector<PrimitiveDev>& dstPrim,
        std::vector<glm::vec3>& dstVec,
        std::vector<glm::vec3>& dstNor,
        std::vector <glm::vec2>& dstUV);

public:
    Scene(std::string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Object> objects;
    std::unordered_map<std::string, uint32_t> MatNameToID;
    std::string skyboxPath;

    // full scene primitives
    std::vector<Primitive> primitives;
    //std::vector<glm::vec3> vertices;
    //std::vector<glm::vec3> normals;
    //std::vector<glm::vec3> uvs;
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
    int hitDepth = 0;

    while (curr != bvhSize) {
        bool hasHit = bvhAABBs[head[curr].bboxID].intersect(r.origin, invDir, tMin);

        if (hasHit)
        {
            ++hitDepth;
            uint32_t s = head[curr].startPrimID;
            uint32_t e = head[curr].endPrimID;
            if (s != UINT32_MAX)
            {
                for (uint32_t i = s; i < e; ++i)
                {
                    float dist;
                    int mid = primitives[i].materialId;
                    uint32_t primId = primNum == triNum ? i :  primitives[i].primId;
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
        glm::vec3 bary;
        triangleIntersection(r, vertices[3 * primID], vertices[3 * primID + 1], vertices[3 * primID + 2], isect.nor, bary);
        if (glm::dot(normals[3 * primID], normals[3 * primID]) > 0.5f)
        {
            isect.nor = normals[3 * primID] * bary.x + normals[3 * primID + 1] * bary.y
                + normals[3 * primID + 2] * bary.z;
        }
        if (uvs)
        {
            isect.uv = uvs[3 * primID] * bary.x + uvs[3 * primID + 1] * bary.y
                + uvs[3 * primID + 2] * bary.z;
        }
        

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

__device__ __inline__ void SceneDev::sampleEnv(PathSegment& segment)
{
    if (envMap != 0)
    {
        
    }
    
}