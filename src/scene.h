#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_gltf.h"
using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
	void loadFromGltf(const std::string& gltfName);
public:
    Scene(string filename);
    ~Scene();

	void parsePrimitive(const tinygltf::Model& model, const tinygltf::Primitive& primitive, Mesh& mesh);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
	std::vector<Mesh> meshes;
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> normals;
	std::vector<Triangle> triangles;
	std::vector<Texture> textures;
	std::vector<glm::vec2> baseColorUvs;
	std::vector<glm::vec2> normalUvs;
	std::vector<glm::vec2> emissiveUvs;
    RenderState state;
};
