#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <fstream>
#include <cstring>
#include <unordered_map>

#include "glm/glm.hpp"
#include "json.hpp"

#include "utilities.h"
#include "sceneStructs.h"
#include "meshLoader.h"

using namespace std;
using json = nlohmann::json;

class Mesh {   
public:
    Mesh();
    ~Mesh();

    std::vector<Triangle> faces;
    std::vector<float> m_cdf;

    // void loadGLTFOrGLB(const std::string &filepath);
    const float computeTriangleArea(const Triangle &t);
    void addTriangleAreaToCDF(const float area);
    void normaliseCDF();
};

class Scene
{
private:
    ifstream fp_in;

    void loadMesh(const std::string &filepath, Mesh &mesh);
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Geom> lights;
    std::vector<Material> materials;
    RenderState state;
};
