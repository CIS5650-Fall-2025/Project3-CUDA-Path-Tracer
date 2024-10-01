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
public:
    Mesh();
    ~Mesh();

    std::vector<Triangle> faces;
    std::vector<float> m_cdf;

    void loadOBJ(const std::string &filepath);
    const float computeTriangleArea(const Triangle &t);
    void addTriangleAreaToCDF(const float area);
    void normaliseCDF();
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
    std::vector<Geom> lights;
    std::vector<Material> materials;
    RenderState state;
};
