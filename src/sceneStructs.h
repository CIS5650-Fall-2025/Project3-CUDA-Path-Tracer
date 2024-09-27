#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "material.h"
#include "mathUtils.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    TRIANGLE,
    SPHERE,
    CUBE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Transformations
{
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Geom
{
    Transformations transforms;
    int materialid;
    enum GeomType type;
};

struct MeshData
{
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
};

struct Object
{
    MeshData meshData;
    Transformations transforms;
    int materialid;
    enum GeomType type;
};

struct AABB
{
    glm::vec3 bmin = glm::vec3(FLT_MAX);
    glm::vec3 bmax = glm::vec3(-FLT_MAX);

    __host__ __device__ inline glm::vec3 center() const;
    __host__ __device__ inline glm::vec3 extend() const;
    __host__ __device__ inline float surfaceArea() const;
    __host__ __device__ inline int majorAxis() const;
    __host__ __device__ inline AABB mergeAABB(const AABB& box) const;
    __host__ __device__ inline void expand(const AABB& box);
    __host__ __device__ inline void expand(const glm::vec3& v);
    __host__ __device__ inline bool intersect(const glm::vec3& ori, const glm::vec3& invDir, float tMin) const;
    __host__ __device__ static inline AABB getAABB(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2);
    __host__ __device__ static inline AABB getAABB(GeomType type, const glm::mat4& transform);

};

struct Primitive
{
    AABB bbox;
    uint32_t primId;
    uint32_t materialId;
};

struct PrimitiveDev
{
    uint32_t materialId;
    uint32_t primId;
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
    uint32_t iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 throughput;
    glm::vec3 radiance;
    uint32_t pixelIndex;
    uint32_t remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  glm::vec3 nor;
  glm::vec2 uv;
  float t;
  uint32_t materialId;
  uint32_t primId;
};


__host__ __device__ inline glm::vec3 AABB::center() const
{
    return (bmin + bmax) * 0.5f;
}

__host__ __device__ inline glm::vec3 AABB::extend() const
{
    return bmax - bmin;
}

__host__ __device__ inline float AABB::surfaceArea() const
{
    glm::vec3 extend = bmax - bmin;
    return 2.f * (extend.x * extend.y + extend.y * extend.z + extend.z * extend.x);
}

__host__ __device__ inline int AABB::majorAxis() const
{
    glm::vec3 extend = bmax - bmin;
    if (extend.x < extend.y) return extend.y > extend.z ? 1 : 2;
    else return extend.x > extend.z ? 0 : 2;
}

__host__ __device__ inline AABB AABB::mergeAABB(const AABB& box) const
{
    return { glm::min(bmin, box.bmin), glm::max(bmax, box.bmax) };
}

__host__ __device__ inline void AABB::expand(const AABB& box)
{
    bmin = glm::min(bmin, box.bmin);
    bmax = glm::max(bmax, box.bmax);
}

__host__ __device__ inline void AABB::expand(const glm::vec3& v)
{
    bmin = glm::min(bmin, v);
    bmax = glm::max(bmax, v);
}

// efficient slab: https://jcgt.org/published/0007/03/04/
__host__ __device__ inline bool AABB::intersect(const glm::vec3& ori, const glm::vec3& invDir, float tMin) const
{
    glm::vec3 t0 = (bmin - ori) * invDir;
    glm::vec3 t1 = (bmax - ori) * invDir;
    glm::vec3 tmin = glm::min(t0, t1), tmax = max(t0, t1);
    float tNear = math::maxComponent(tmin);
    float tMax = math::minComponent(tmax);
    //return tNear <= math::minComponent(tmax);
    return tMax > 0.f && tNear < tMin && tNear <= tMax;
}

__host__ __device__ inline AABB AABB::getAABB(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
{
    glm::vec3 bmin = glm::min(glm::min(v0, v1), v2);
    glm::vec3 bmax = glm::max(glm::max(v0, v1), v2);
    return { bmin, bmax };
}

__host__ __device__ inline AABB AABB::getAABB(GeomType type, const glm::mat4& transform)
{
    if (type == CUBE)
    {
        glm::vec4 verts[8] = {
            glm::vec4(0.5f, 0.5f, 0.5f, 1.f),
            glm::vec4(0.5f, 0.5f, -0.5f, 1.f),
            glm::vec4(0.5f, -0.5f, 0.5f, 1.f),
            glm::vec4(0.5f, -0.5f, -0.5f, 1.f),
            glm::vec4(-0.5f, 0.5f, 0.5f, 1.f),
            glm::vec4(-0.5f, 0.5f, -0.5f, 1.f),
            glm::vec4(-0.5f, -0.5f, 0.5f, 1.f),
            glm::vec4(-0.5f, -0.5f, -0.5f, 1.f)
        };

        glm::vec3 bmin = glm::vec3(FLT_MAX);
        glm::vec3 bmax = glm::vec3(FLT_MIN);

        for (auto& v : verts)
        {
            v = transform * v;
            bmin = glm::min(bmin, glm::vec3(v));
            bmax = glm::max(bmax, glm::vec3(v));
        }
        return { bmin, bmax };
    }
    else if (type == SPHERE)
    {
        glm::vec4 verts[6] = {
            glm::vec4(0.5f, 0.f, 0.f, 1.f),
            glm::vec4(-0.5f, 0.f, 0.f, 1.f),
            glm::vec4(0.f, 0.5f, 0.f, 1.f),
            glm::vec4(0.f, -0.5f, 0.f, 1.f),
            glm::vec4(0.f, 0.f, 0.5f, 1.f),
            glm::vec4(0.f, 0.f, -0.5f, 1.f),
        };

        glm::vec3 bmin = glm::vec3(FLT_MAX);
        glm::vec3 bmax = glm::vec3(FLT_MIN);

        for (auto& v : verts)
        {
            v = transform * v;
            bmin = glm::min(bmin, glm::vec3(v));
            bmax = glm::max(bmax, glm::vec3(v));
        }
        return { bmin, bmax };
    }
    
    return AABB();
}