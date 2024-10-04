#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include <tiny_gltf.h>

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string &jsonName);
    void loadFromGltf(const std::string &gltfName);

    void setupCamera(
        glm::ivec2 resolution = glm::ivec2(800, 800),
        glm::vec3 position = glm::vec3(0, 0, 10),
        glm::vec3 lookAt = glm::vec3(0, 0, 0),
        float fovy = 45,
        glm::vec3 up = glm::vec3(0, 1, 0),
        float lensSize = 0,
        float focalDist = 0);

    void loadGltfMaterial(const tinygltf::Model &model, int materialId);
    template <typename index_t>
    void loadGltfTriangles(size_t count, const index_t *indices, const glm::vec3 *positions);
    void loadGltfMesh(const tinygltf::Model &model, int meshId);
    void loadGltfNode(const tinygltf::Model &model, int node);

public:
    Scene(string filename);
    ~Scene() = default;

    std::vector<Geom> geoms;
    std::vector<Mesh> meshes;
    std::vector<Tri> tris;
    size_t numLights;
    std::vector<Material> materials;
    RenderState state;
    bool useDirectLighting;
};
