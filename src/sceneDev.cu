#include "sceneDev.h"


__device__ bool SceneDev::intersect(const Ray& r, ShadeableIntersection& isect)
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
                    uint32_t primId = primNum == triNum ? i : primitives[i].primId;
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


__device__ bool SceneDev::visibilityTest(const Ray& r, float dist)
{
    float tMin = dist;

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
                    float d;
                    uint32_t primId = primNum == triNum ? i : primitives[i].primId;
                    bool hit = intersectPrimitives(r, primId, d);

                    if (hit && d < tMin)
                    {
                        return false;
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

    return true;
}

__device__ bool SceneDev::intersectPrimitives(const Ray& r, uint32_t primID, float& dist)
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


__device__ void SceneDev::intersectPrimitivesDetail(const Ray& r, uint32_t primID, ShadeableIntersection& isect)
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
            isect.uv.y = 1.f - isect.uv.y;
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



__device__  glm::vec3 SceneDev::sampleEnv(const glm::vec3& ori, glm::vec3& wi, glm::vec3 rng, float* pdf)
{
    uint32_t idx = rng.x * envMapWidth * envMapHeight;
    idx = envMapDistrib[idx].cdfID;

    uint32_t y = idx / envMapWidth;
    uint32_t x = idx - y * envMapWidth;
    glm::vec2 uv = glm::vec2((.5f + x) / envMapWidth, (.5f + y) / envMapHeight);

    float4 skyCol4 = tex2D<float4>(envMap, uv.x, uv.y);
    glm::vec3 skyColor = glm::vec3(skyCol4.x, skyCol4.y, skyCol4.z);

    wi = glm::normalize(math::planeToDir(uv));
    Ray r = { ori + EPSILON * wi , wi };
    bool vis = visibilityTest(r, FLT_MAX);

    if (!vis)
    {
        *pdf = 0.f;
        return glm::vec3(0);
    }
    else
    {
        *pdf = math::luminance(skyColor) * envMapPdfSumInv * envMapWidth * envMapHeight * INV_PI * INV_PI * 0.5f;
        return skyColor;
    }
}

__device__ glm::vec3 SceneDev::getEnvColor(const glm::vec3& dir)
{
    glm::vec2 uv = math::sampleSphericalMap(dir);
    float4 skyCol4 = tex2D<float4>(envMap, uv.x, uv.y);
    return glm::vec3(skyCol4.x, skyCol4.y, skyCol4.z);
}

__device__ float SceneDev::envMapPdf(const glm::vec3& wi)
{
    glm::vec2 uv = math::sampleSphericalMap(wi);
    float4 skyCol4 = tex2D<float4>(envMap, uv.x, uv.y);
    glm::vec3 skyColor = glm::vec3(skyCol4.x, skyCol4.y, skyCol4.z);
    return math::luminance(skyColor) * envMapPdfSumInv * envMapWidth * envMapHeight * 0.5f;
}