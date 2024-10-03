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
    void loadFromJSON(const std::string& jsonName);
    void loadFromGltf(const std::string& gltfName);

    void loadGltfMaterial(const tinygltf::Model &model, int materialId);
    template<typename index_t>
    void loadGltfTriangles(size_t count, const index_t *indices, const glm::vec3 *positions);
    void loadGltfMesh(const tinygltf::Model &model, int meshId);
    void loadGltfNode(const tinygltf::Model& model, int node);
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
