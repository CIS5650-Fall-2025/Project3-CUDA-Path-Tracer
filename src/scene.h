#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    bool loadGLTF(const std::string& filename, int material, const glm::vec3 translation, const glm::vec3 rotation, const glm::vec3 scale);

public:
    Scene(string filename);
    ~Scene() = default;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    RenderState state;
};
