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

//Add as triangle
struct Triangle
{
	//glm::vec3 v0;
	//glm::vec3 v1;
	//glm::vec3 v2;
 //   //for texture
 //   glm::vec2 uv0;
 //   glm::vec2 uv1;
 //   glm::vec2 uv2;
	////for normal
	//glm::vec3 n0;
 //   glm::vec3 n1;
 //   glm::vec3 n2;
    glm::vec3 verts[3];
    glm::vec2 uvs[3];
    glm::vec3 normals[3];
    //tangent
    glm::vec3 tangent;
    //bitangent
    glm::vec3 bitangent;
};

struct Texture
{
	unsigned char* data;
	int width;
	int height;
	int channels;
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
    int triIndexStart;
    int triIndexEnd;
    //Add for texture
    int textureid = -1 ;
    int hasTexture = 0 ;
    //Add for normal
    int normalid = -1;
    int hasNormal = 0;
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

    //Add for texture

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

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  bool outside;
  int textureid = -1;
  glm::vec2 uv;
  int normalid = -1;
  //tangent
  glm::vec3 tangent;
  //bitangent
  glm::vec3 bitangent;
};
