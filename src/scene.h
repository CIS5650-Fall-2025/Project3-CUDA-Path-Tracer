#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "BVH.h"
#include "texture.h"

using namespace std;



enum GeomType
{
    SPHERE,
    CUBE,
    MESH
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


    int num_BVHNodes;
    int num_triangles;
    BVHNode* bvhNodes;
    Triangle* triangles;
};


class Scene {
private:
    std::ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    void LoadFromOBJ(const std::string& fileName, Geom& geom);

public:
    Scene(std::string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<TextureData> textureData;   // CPU-side images (hostPixels, etc.)
    std::vector<TextureInfo> textureInfo;      // GPU-side texture objects
};