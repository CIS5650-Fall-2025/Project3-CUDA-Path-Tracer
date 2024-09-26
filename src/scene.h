#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_obj_loader.h"
#include "texture.h"
#include "cudaUtilities.h"

using namespace std;
using namespace tinyobj;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
	std::vector<Triangle> triangles;
	Texture* envMap;
    RenderState state;
    void createCube(uint32_t materialid, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
	void createSphere(Geom& sphere, uint32_t materialid, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
	void loadObj(const std::string& filename, uint32_t materialid = 0, glm::vec3 translation = glm::vec3(0), glm::vec3 rotation = glm::vec3(0), glm::vec3 scale = glm::vec3(1.));
	void addMaterial(Material& m);
    void loadEnvMap(const char* filename);
    static void updateTransform(Geom& geom, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
	static void updateTransform(Geom& geom, Triangle* triangles);
};
