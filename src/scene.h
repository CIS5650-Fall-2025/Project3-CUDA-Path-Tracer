#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <fstream>

#include "glm/glm.hpp"
#include "tinyobjloader/tiny_obj_loader.h"

#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Mesh {
private:
    std::vector<Triangle> faces;
public:
    Mesh();
    ~Mesh();

    void loadOBJ(const std::string &filepath);
    const std::vector<Triangle> &getFaces();
};

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
    RenderState state;
};
