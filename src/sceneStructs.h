#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    // declare the mesh geometry type
    MESH,

    SPHERE,
    CUBE
};

// declare the struct that stores the texture data
struct texture_data {

    // declare the variable for the width of the texture
    int width {0};

    // declare the variable for the height of the texture
    int height {0};

    // declare the variable for the index of the first pixel
    int index {0};
};

// declare the struct that stores the vertex data
struct vertex_data {

    // declare the variable for the material index
    int material_index {-1};

    // declare the variable for the location of the vertex
    glm::vec3 point {glm::vec3(0.0f)};

    // declare the variable for the normal of the vertex
    glm::vec3 normal {glm::vec3(0.0f)};

    // declare the variable for the tangent of the vertex
    glm::vec3 tangent {glm::vec3(0.0f)};

    // declare the variable for the texture coordinate of the vertex
    glm::vec2 coordinate {glm::vec2(0.0f)};
};

// declare the struct that stores the bounding sphere generation data
struct bounding_sphere_generation_data {

    // declare the variable for the center of the bounding sphere
    glm::vec3 center {glm::vec3(0.0f)};

    // declare the variable for the radius of the bounding sphere
    float radius {0.0f};

    // declare the variable for the indices of the bounding sphere's children
    int child_indices[2] {-1, -1};

    // declare the variable for the vertices inside this bounding sphere
    std::vector<vertex_data> vertices {};
};

// declare the struct that stores the bounding sphere data
struct bounding_sphere_data {

    // declare the variable for the center of the bounding sphere
    glm::vec3 center {glm::vec3(0.0f)};

    // declare the variable for the radius of the bounding sphere
    float radius {0.0f};

    // declare the variable for the indices of the bounding sphere's children
    int child_indices[2] {-1, -1};

    // declare the variable for the index of the first vertex inside this bounding sphere
    int index {-1};

    // declare the variable for the number of triangles inside this bounding sphere
    int count {0};
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
};

struct Material
{
    // declare the variable for the index of the diffuse texture
    int diffuse_texture_index {-1};

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

  // declare the variable for the texture coordiante at the intersection
  glm::vec2 coordiante;
};
