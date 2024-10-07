#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define PI                3.1415926535897932384626422832795028841971f

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE
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
    bool has_motion;
    glm::vec3 velocity;

    __device__ __host__
    static glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale)
    {
        glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
        glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
        rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
        rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
        glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
        return translationMat * rotationMat * scaleMat;
    }

    __device__ __host__
    void update(float dT)
    {
        if (!this->has_motion) { return; }
        glm::vec3 new_translation = translation + dT * velocity;
        this->transform = this->buildTransformationMatrix(
            new_translation, this->rotation, this->scale);
        this->inverseTransform = glm::inverse(this->transform);
        this->invTranspose = glm::inverseTranspose(this->transform);
    }
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
    float roughness;
    struct
    {
        bool hasDispersion;
        float indexOfRefraction[3];
    } dispersion;
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
    float aperture;
    float exposure;

    void serialize(std::ofstream& ofs);
    void deserialize(std::ifstream& ifs);
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;

    void serialize(std::ofstream& ofs);
    void deserialize(std::ifstream& ifs);
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
  unsigned char materialId;
};
