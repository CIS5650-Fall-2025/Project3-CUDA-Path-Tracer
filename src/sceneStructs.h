#pragma once

#include <string>
#include <vector>
#include <array>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE = -3,
    CUBE = -2,
    SQUARE = -1
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    union
    {
        enum GeomType type;
        int meshId;
    };
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    Geom() : scale(1) {};
};

struct Mesh
{
    int triCount;
    int indOffset;
    int pointOffset;
    int uvOffset;

    Mesh() : triCount(0), indOffset(0), uvOffset(-1) {};
};

struct TextureData
{
    glm::ivec2 dimensions;
    std::vector<glm::vec4> data;
};

union Texture
{
    glm::vec3 value;
    struct
    {
        int32_t negTextureIndex;
        // Assumption: a cudaTextureObject_t is 8 bytes (should be on most platforms)
        cudaTextureObject_t textureHandle;
    };
    Texture() : value(glm::vec3(1, 0, 1)) {}
    Texture(const Texture& other) {
        if (other.negTextureIndex < 0) {
            negTextureIndex = other.negTextureIndex;
            textureHandle = other.textureHandle;
        } else {
            value = other.value;
        }
    }
};

struct Material
{
    Texture albedo;
    glm::vec3 emittance;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;

    Material()
        : hasReflective(false), hasRefractive(false), indexOfRefraction(1.55f), emittance(0.f)
    {
        albedo.value = glm::vec3(1);
    }
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
    float lensSize;
    float focalDist;

    Camera() = default;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;

    RenderState() : camera(), iterations(5000), traceDepth(8), imageName("out_image") {}
};

struct PathSegment
{
    Ray ray;
    glm::vec3 throughput;
    glm::vec3 radiance;
    int pixelIndex;
    int remainingBounces;
};

struct PathActive
{
    __host__ __device__ bool operator()(const PathSegment &path)
    {
        return path.remainingBounces != 0;
    }
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    float t;
    glm::vec3 surfaceNormal;
    int materialId;
    glm::vec2 uv;
};