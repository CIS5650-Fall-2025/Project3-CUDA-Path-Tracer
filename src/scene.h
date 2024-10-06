#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "glTFLoader.h"
#include <unordered_map>

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    bool jsonLoadedNonCuda = false;
    std::string jsonName_str;

    void loadFromJSON(const std::string& jsonName);
    int triangleCount = -1;
    std::vector<MeshTriangle> triangles;
public:
    Scene(string filename);
    ~Scene(){};

    std::vector<MeshTriangle> getTriangleBuffer();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
